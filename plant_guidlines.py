from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PlantScienceGuidelines:
    def __init__(self):
        self.guidelines = self._load_guidelines()

    # Returns a hard-coded list of useful botanical rules, tips, and criteria
    # (e.g., signs of common diseases, prevention tips, and when to take action).
    def _load_guidelines(self) -> List[str]:
        return [
            "Powdery mildew often appears as white, powdery spots on leaves and stems.",
            "To prevent fungal diseases, ensure good air circulation around plants and avoid overhead watering.",
            "Yellowing leaves (chlorosis) can indicate nutrient deficiencies, particularly nitrogen.",
            "Blight often causes sudden browning, blackening, and death of plant tissues like leaves and flowers.",
            "Crop rotation is a key strategy to prevent soil-borne diseases from building up.",
            "Rust diseases typically present as small, reddish-brown to orange pustules on the underside of leaves.",
            "Aphids and spider mites are common pests that can cause leaf curling, stippling, and transmit viruses.",
            "A sudden wilt on a sunny day, even with moist soil, can be a sign of root rot or vascular disease.",
            "Necrotic spots (dead tissue) that are brown or black can be a symptom of bacterial or fungal infections.",
            "Always remove and destroy infected plant debris to reduce the spread of pathogens.",
            "Fungicides are most effective when applied preventatively, before disease symptoms are widespread.",
            "Wilting, accompanied by dark streaks inside the stem, suggests a vascular wilt disease like Fusarium or Verticillium.",
            "Mosaic viruses cause mottled patterns of light green, yellow, and dark green on leaves."
        ]
    # Returns the full list of plant science guidelines for reference or display.
    def get_all_guidelines(self) -> List[str]:
        return self.guidelines

    # Looks for and returns only those guidelines that contain a given keyword (case-insensitive),
    # making it easier to find relevant advice (for example, all rules mentioning "fungal" or "mildew").
    def search_guidelines_by_keyword(self, keyword: str) -> List[str]:
        keyword_lower = keyword.lower()
        return [g for g in self.guidelines if keyword_lower in g.lower()]
