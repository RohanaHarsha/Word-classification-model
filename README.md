# Word Classification Model for Real Estate Search

## Overview
This model enhances search functionality by allowing users to enter natural language queries. It extracts relevant keywords related to real estate and housing, ensuring more accurate and reliable search results.

## Features
- Converts user queries into structured keywords.
- Enhances real estate search accuracy.
- Can be improved by modifying the dataset.

## Setup and Usage
### Installation
```sh
pip install -r requirements.txt
```

### Running the Model
```sh
python main.py
python nlp_service.py
```

### Improving Accuracy
Modify `models.py` and replace hardcoded sentences with a CSV dataset. The extracted results will be saved in `nlp_model.py` and `tokenizer.pickle`.

## File Structure
```
.
├── main.py           # Initializes the model
├── nlp_service.py    # Extracts keywords from queries
├── models.py         # Model logic, can be modified for better accuracy
├── tokenizer.pickle  # Stores tokenized data for processing
├── requirements.txt  # Required dependencies
└── README.md         # Project documentation
```

## Git Setup
```sh
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

## Contributing
Feel free to modify and improve the model by enhancing the dataset and optimizing keyword extraction.

## License
This project is open-source. Feel free to use and modify it as needed.

