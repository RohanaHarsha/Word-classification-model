from nlp_service import nlpService

if __name__ == "__main__":
    nlp_service = nlpService()
    
    # Simplified query for better keyword extraction
    query = "Looking for a house with modern amenities in a neighborhood with good schools and parks."

    # Extract relevant properties
    properties = nlp_service.search_properties(query)
    
    # Output the results
    print("Relevant properties:", properties)
