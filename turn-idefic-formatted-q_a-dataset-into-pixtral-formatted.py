#this takes the q & a csv & downloaded_images and maps them to the format needed 
#for fine-tuning eg pixtral model
#could use this for other formats too, just make sure to map things correctly in the create_messages json template
import pandas as pd
from datasets import Dataset, Image
import os
from typing import List, Dict, Any

def format_qa_dataset(
    csv_path: str,
    images_dir: str
) -> Dataset:
    """
    Format CSV data and images into a dataset suitable for vision Q&A fine-tuning.
    
    Args:
        csv_path: Path to CSV file containing 'id', 'query', and 'answers' columns
        images_dir: Directory containing images where filenames match the 'id' column
        
    Returns:
        datasets.Dataset: Formatted dataset with 'messages' and 'images' columns
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    def create_messages(row: pd.Series) -> List[Dict[Any, Any]]:
        """Create formatted messages list for a single row."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": row['query'],
                        "index": None
                    },
                    {
                        "type": "image",
                        "text": None,
                        "index": 0
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": row['answers'],
                        "index": None
                    }
                ]
            }
        ]
    
    def load_image(image_id: str) -> List[str]:
        """Get full path for image file, returning as a list."""
        return [os.path.join(images_dir, f"{image_id}.jpg")] #<- pay attention file extension
    
    # Create formatted data
    formatted_data = {
        'messages': df.apply(create_messages, axis=1).tolist(),
        'images': df['id'].apply(load_image).tolist()
    }
    
    # Create dataset
    dataset = Dataset.from_dict(formatted_data)
    
    # Cast images column to Image type
    dataset = dataset.cast_column("images", [Image()])
    
    return dataset

dataset2 = format_qa_dataset(
    csv_path='/content/artemis_cup_theseus_qa_pairs_filtered.csv',
    images_dir='downloaded_images'
)

split_dataset2 = dataset2.train_test_split(test_size=TEST_SIZE, shuffle=False)
