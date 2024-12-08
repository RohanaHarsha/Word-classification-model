from nlp_module import NLPModule
class nlpService:
    def __init__(self):
        self.nlp_module = NLPModule()
    
    def search_properties(self, query: str) -> list:
        # Extract keywords using the NLP module
        keywords = self.nlp_module.predict(query)
        if keywords:
            print(f"Extracted keywords: {keywords}")
            # Use the extracted keywords to search for properties
            return self.get_properties_by_keywords(keywords)
        else:
            print("No relevant keywords found.")
            return []

    def get_properties_by_keywords(self, keywords: list) -> list:
        # Example property listings
        example_properties = [
            {'id': 1, 'description': 'House with modern kitchen and large backyard.'},
            {'id': 2, 'description': 'Loft apartment with exposed brick walls.'},
            {'id': 3, 'description': 'Penthouse with panoramic city views and rooftop terrace.'},
            {'id': 4, 'description': 'Townhouse with access to swimming pool and garden area.'},
        ]

        # Search for properties that match any of the extracted keywords
        matching_properties = []
        for prop in example_properties:
            for keyword in keywords:
                if keyword in prop['description'].lower():
                    matching_properties.append(prop)
                    break  # Avoid adding the same property multiple times
        
        return matching_properties