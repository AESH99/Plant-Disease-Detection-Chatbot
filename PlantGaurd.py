import streamlit as st
import torch
from PIL import Image
import numpy as np
import logging
# Updated import to use the plant-focused multimodal processor
from plant_multimodal_processor import MultimodalPlantProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PlantGuard - Plant Disease Assistant",
    page_icon="🌿", # Updated icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated styling for a green, plant-themed header
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

@st.cache_resource
def load_processor():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Loading PlantGuard models on {device.upper()}...")
        # Instantiate the correct processor
        processor = MultimodalPlantProcessor(device=device)
        st.success("PlantGuard models loaded successfully!")
        return processor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def reset_analysis():
    st.session_state.analysis_results = None
    st.rerun()

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌿 PlantGuard: Multimodal Plant Disease Assistant</h1>
        <p>AI-powered preliminary plant health assessment using image, audio, and text analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Updated disclaimer for agricultural/botanical context
    st.warning("""
    ⚠️ **DISCLAIMER**: This tool provides preliminary assessments for educational purposes only and should NOT replace professional agronomic advice.
    - Always consult qualified agricultural extension agents or plant pathologists for accurate diagnosis and treatment.
    - Widespread or severe symptoms require immediate professional evaluation.
    - The AI may not detect all conditions or provide complete assessments.
    """)

    if st.session_state.processor is None:
        st.session_state.processor = load_processor()

    if st.session_state.processor is None:
        st.error("Failed to load PlantGuard models. Please refresh the page.")
        return

    # Sidebar updated for plant context instead of patient info
    with st.sidebar:
        st.header("🌱 Plant & Context")

        species = st.text_input("Plant Species", placeholder="e.g., Tomato, Corn, Apple")
        location = st.selectbox("Location / Environment", ["Not specified", "Outdoor Field", "Greenhouse", "Indoor Pot"])

        st.subheader("📋 Growth History")
        growth_history = st.text_area(
            "Recent treatments or observations",
            placeholder="e.g., Recently applied fertilizer, noticed pests, unusual weather.",
            height=80
        )

        context_info = {
            'species': species if species else None,
            'location': location if location != "Not specified" else None,
            'growth_history': growth_history if growth_history else None
        }

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📷 Leaf Image")

        uploaded_image = st.file_uploader(
            "Upload photo of the affected leaf",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Take a clear, well-lit photo of the leaf showing symptoms."
        )

        image_data = None
        if uploaded_image is not None:
            image_data = Image.open(uploaded_image)
            st.image(image_data, caption="Uploaded leaf image", use_container_width=True)

            if st.session_state.processor:
                quality = st.session_state.processor.image_processor.analyze_image_quality(image_data)

                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    st.metric("Image Quality", quality.get('quality', 'unknown').title())
                with col_q2:
                    st.metric("Resolution", f"{image_data.size[0]}x{image_data.size[1]}")

                if quality.get('issues'):
                    st.warning(f"⚠️ Image issues: {', '.join(quality['issues'])}")

    with col2:
        st.header("🎙️ Audio & Text Input")

        st.subheader("Audio Description")
        uploaded_audio = st.file_uploader(
            "Upload audio description of symptoms",
            type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
            help="Record yourself describing the symptoms, how they've spread, etc."
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')

        st.subheader("📝 Symptom Description")
        text_symptoms = st.text_area(
            "Describe the symptoms in detail",
            placeholder="e.g., When did you first notice this? Are the spots spreading? Are leaves turning yellow?",
            height=120
        )

        # Updated select boxes for plant context
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            duration = st.selectbox(
                "Symptom Duration",
                ["A few days", "About a week", "2-3 weeks", "More than a month"]
            )

        with col_d2:
            changes = st.selectbox(
                "Recent Trend",
                ["No changes", "Spreading/Worsening", "Improving", "New symptoms appearing"]
            )

    st.markdown("---")

    if st.button("🔬 Analyze Plant Health", type="primary", use_container_width=True):
        has_image = uploaded_image is not None
        has_audio = uploaded_audio is not None
        has_text = bool(text_symptoms.strip())

        if not any([has_image, has_audio, has_text]):
            st.error("Please provide at least one input: image, audio, or text description.")
        else:
            combined_text = f"Symptoms: {text_symptoms}. Duration: {duration}. Trend: {changes}"

            with st.spinner("🔍 Analyzing your case using AI models..."):
                try:
                    results = st.session_state.processor.process_case(
                        image=image_data,
                        audio=uploaded_audio,
                        text_data=combined_text,
                        context_info=context_info
                    )

                    st.session_state.analysis_results = results
                    st.success("✅ Analysis completed!")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        if results.get('error'):
            st.error(f"Analysis Error: {results['message']}")
        else:
            st.markdown("## 📊 Analysis Results")

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                # Changed from Urgency to Severity
                severity = results.get('severity', 'unknown')
                severity_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                st.metric("Severity Level", f"{severity_color} {severity.title()}")

            with col_m2:
                confidence = results.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")

            with col_m3:
                # --- ERROR FIX ---
                # Corrected logic to safely check for available modalities,
                # preventing an error if one of the analysis results is None.
                modalities = sum([
                    bool(results.get('image_analysis') and results.get('image_analysis').get('available')),
                    bool(results.get('audio_analysis') and results.get('audio_analysis').get('available')),
                    bool(results.get('text_analysis') and results.get('text_analysis').get('available'))
                ])
                st.metric("Data Sources", f"{modalities}/3")

            # Updated to botanical_assessment and its keys
            if results.get('botanical_assessment'):
                assessment = results['botanical_assessment']

                st.subheader("🌿 Botanical Assessment")
                st.info(assessment.get('symptom_summary', 'No description available'))

                if assessment.get('key_observations'):
                    st.subheader("🔍 Key Observations")
                    for finding in assessment['key_observations']:
                        st.write(f"• {finding}")

                if assessment.get('potential_diseases'):
                    st.subheader("🎯 Potential Issues")
                    for diagnosis in assessment['potential_diseases']:
                        likelihood = diagnosis.get('likelihood', 'unknown')
                        condition = diagnosis.get('condition', 'Unknown condition')

                        likelihood_emoji = {'high': '🔴', 'moderate': '🟡', 'low': '🟢'}.get(likelihood, '⚪')
                        st.write(f"{likelihood_emoji} **{condition}** - {likelihood.title()} likelihood")

            if results.get('recommendations'):
                st.subheader("💡 Recommendations")
                for rec in results['recommendations']:
                    st.write(f"• {rec}")

            # Updated final warning
            st.warning("""
            ⚠️ **Important Reminders:**
            - This is a preliminary assessment tool only.
            - Always consult with qualified agricultural professionals for accurate diagnosis.
            - Seek immediate professional advice if the condition is spreading rapidly.
            """)

            st.markdown("---")
            col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 1])
            with col_reset2:
                if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
                    reset_analysis()

    st.markdown("---")
    # Updated footer
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        PlantGuard - Multimodal Plant Disease Assistant<br>
        Powered by BLIP, Whisper, and Sentence Transformers<br>
        <strong>For educational and preliminary assessment purposes only</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

