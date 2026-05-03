import json
import traceback

def extract_notebook(notebook_path, output_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    f.write("".join(cell.get('source', [])) + "\n\n")
        print(f"Extraction successful: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    extract_notebook(
        '/Users/kaysarulanasapurba/Desktop/ResumeProjects/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images/notebooks/training.ipynb',
        '/Users/kaysarulanasapurba/Desktop/ResumeProjects/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images/src/train_extracted.py'
    )
