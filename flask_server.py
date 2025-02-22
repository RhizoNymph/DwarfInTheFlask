import os
import re
import io
import json
import base64
import logging
import hashlib
import subprocess
import tempfile
import imghdr

import torch
#import librosa
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel
from ragatouille import RAGPretrainedModel
from claudette import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

# Single state file
STATE_FILE = 'state.json'

def load_state():
    """Load the state from a single JSON file"""
    default_state = {
        "document_mapping": {
            "next_id": 0,
            "mappings": {}
        },
        "document_metadata": {},
        "indexed_files": {},
        "file_indices": [],
        "indexed_texts": {},
        "text_indices": []
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return default_state
    return default_state

def save_state(state):
    """Save the state to a single JSON file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# Initialize state
state = load_state()

@app.route('/get_indices', methods=['GET'])
def get_indices():
    return jsonify(state["file_indices"])

def validate_image(base64_data):
    try:
        image_bytes = base64.b64decode(base64_data)
        image_type = imghdr.what(None, h=image_bytes)
        if image_type:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            return image_type
        else:
            return None
    except Exception as e:
        print(f"Image validation failed: {e}")
        return None

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def calculate_text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def extract_metadata_with_claude(file_path):
    content = "Based on the following pages from the document, what is the title of the document and who are its authors? Please respond in JSON format with 'title' and 'authors' keys.\n\n"
    processed_messages = [{"type": "text", "text": content}]

    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(
            file_path,
            first_page=1,
            last_page=10,
            thread_count=os.cpu_count() - 1,
            output_folder=path,
        )

        for i, image in enumerate(images):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            processed_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_str
                }
            })

    chat = Chat("claude-3-5-sonnet-20240620")
    response = chat(processed_messages)

    try:
        metadata = json.loads(response.content[0].text)
        return metadata.get('title', ''), metadata.get('authors', '')
    except json.JSONDecodeError:
        return '', ''

def extract_metadata_with_qwen(file_path):
    # Initialize model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    with tempfile.TemporaryDirectory() as path:
        # Convert PDF pages to images
        images = convert_from_path(
            file_path,
            first_page=1,
            last_page=5,
            thread_count=os.cpu_count() - 1,
            output_folder=path,
        )

        # Save images and create content list
        content = []
        for i, image in enumerate(images):
            temp_image_path = os.path.join(path, f"page_{i}.png")
            image.save(temp_image_path)
            content.append({
                "type": "image",
                "image": f"file://{temp_image_path}",
            })

        # Add the text prompt
        content.append({
            "type": "text",
            "text": "What is the title of this paper? Who are the authors? Please respond in a json format with the keys Title and Authors. Authors should just be a list of names."
        })

        messages = [{
            "role": "user",
            "content": content
        }]

        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(output_text)
        try:
            json_str = output_text
            if '```json' in json_str:
                json_str = json_str.split('```json\n')[1].split('\n```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```\n')[1].split('\n```')[0]

            metadata = json.loads(json_str)
            title = metadata.get('Title', '')
            authors = metadata.get('Authors', [])
            if isinstance(authors, list):
                authors = ', '.join(authors)
            return title, authors
        except Exception as e:
            print(f"Error parsing Qwen output: {output_text}")
            print(f"Error details: {str(e)}")
            return '', ''

    return '', ''

@app.route('/get_indexed_hashes', methods=['GET'])
def get_indexed_hashes():
    return jsonify(state["indexed_files"])

@app.route('/get_document_metadata', methods=['GET'])
def get_document_metadata():
    return jsonify(state["document_metadata"])

@app.route('/get_document_mapping', methods=['GET'])
def get_document_mapping():
    return jsonify(state["document_mapping"]["mappings"])

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    model = data.get('model', 'claude-3-5-sonnet-20240620')
    chat = Chat(model)

    processed_messages = []
    for message in messages:
        if message['type'] == 'image_url':
            image_data = message['image_url']['url']
            image_type = validate_image(image_data)
            if not image_type:
                return jsonify({"error": "Invalid image data"}), 400
            media_type = f"image/{image_type}"
            processed_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
        elif message['type'] == 'text':
            processed_messages.append({"type": "text", "text": message['text']})

    response = chat(processed_messages)

    formatted_response = {
        "choices": [{
            "message": {
                "content": response.content[0].text if response.content else "",
                "role": "assistant"
            }
        }],
        "usage": {
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
        }
    }

    return jsonify(formatted_response)

@app.route('/search', methods=['POST'])
def search():
    global state  # Only use the state variable
    state = load_state()  # Load the latest state before performing the search
    data = request.json
    query = data.get('query', "")
    if query == "":
        return "Search query required"
    index_name = data.get("index_name", "universal")
    print(index_name)
    topk = data.get('topk', 10)
    image_results = []
    text_results = []

    if os.path.exists(f'./.byaldi/{index_name}'):
        MultiRAG = RAGMultiModalModel.from_index(f"{index_name}", verbose=1)
        MultiRAG.model.model = MultiRAG.model.model.float()
        raw_image_results = MultiRAG.search(query, k=topk)
        for result in raw_image_results:
            # Use state directly instead of document_mapping variable
            doc_id_str = str(result.doc_id)
            metadata = state["document_mapping"]["mappings"].get(doc_id_str, {})

            processed_result = {
                "score": float(result.score),
                "doc_id": doc_id_str,
                "page_num": str(result.page_num) if hasattr(result, 'page_num') else None,
                "metadata": {k: str(v) for k, v in result.metadata.items()} if result.metadata else {},
                "base64": result.base64,
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "filename": metadata.get("filename", "Unknown")
            }
            image_results.append(processed_result)

    if os.path.exists(f'./.ragatouille/colbert/indexes/{index_name}'):
        TextRAG = RAGPretrainedModel.from_index(f"./.ragatouille/colbert/indexes/{index_name}", verbose=1)
        raw_text_results = TextRAG.search(query, k=topk)

        for result in raw_text_results:
            doc_id_str = str(result.get('document_id'))
            # Use state directly instead of document_mapping variable
            metadata = state["document_mapping"]["mappings"].get(doc_id_str, {})

            processed_result = {
                "content": result.get('content'),
                "score": float(result.get('score')),
                "doc_id": doc_id_str,
                "page_num": str(result.get('page_num')) if hasattr(result, 'page_num') else None,
                "metadata": metadata,
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "filename": metadata.get("filename", "Unknown")
            }
            text_results.append(processed_result)

    return jsonify({"images": image_results, "texts": text_results})

@app.route('/indexPDF', methods=['POST'])
def index_pdf():
    logger.info("Starting index_pdf function")
    global state
    indexed_files = state["indexed_files"]
    indexed_texts = state["indexed_texts"]
    document_mapping = state["document_mapping"]
    document_metadata = state["document_metadata"]
    data = request.json

    if not data or 'pdf_content' not in data or 'filename' not in data:
        logger.error("Missing PDF content or filename")
        return jsonify({'error': 'Missing PDF content or filename'}), 400

    filename = secure_filename(data['filename'])
    pdf_content = base64.b64decode(data['pdf_content'])
    index_name = data.get('index_name', 'universal')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_content)
        file_path = temp_file.name

    file_hash = calculate_sha256(file_path)

    mapped = False
    doc_id = None
    title = None
    authors = None
    # Check if already indexed
    for current_doc_id, mapping in state["document_mapping"]["mappings"].items():
        if mapping['hash'] == file_hash:
            mapped = True
            doc_id = int(current_doc_id)
            title = mapping["title"]
            authors = mapping["authors"]

    if mapped == False:
        # Get next doc_id
        doc_id = state["document_mapping"]["next_id"]
        state["document_mapping"]["next_id"] += 1

        # Extract metadata
        try:
            title, authors = extract_metadata_with_qwen(file_path)
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            title, authors = "Unknown", "Unknown"

        # Update state
        state["document_mapping"]["mappings"][doc_id] = {
            'hash': file_hash,
            'title': str(title),
            'authors': str(authors),
            'filename': filename
        }
        save_state(state)  # Save state after updating document_mapping

        metadata_dict = {
            doc_id: {
                "title": str(title),
                "authors": str(authors),
                "filename": str(filename),
                "doc_id": str(doc_id)
            }
        }

    if file_hash in indexed_files:
        if index_name in indexed_files[file_hash]:
            os.remove(file_path)
            return jsonify({'message': 'File already indexed', 'doc_id': doc_id})

    try:
        logger.info("Creating new individual index")
        if not os.path.exists(f'./.byaldi/{file_hash}'):
            RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
            RAG.model.model = RAG.model.model.float()
            RAG.index(
                input_path=file_path,
                index_name=file_hash,
                store_collection_with_index=True,
                overwrite=False,
                doc_ids=[doc_id]
            )
    except Exception as e:
        logger.exception(f"Error indexing PDF {filename}: {str(e)}")
        return jsonify({'error': 'Failed to index PDF', 'details': str(e)}), 500

    logger.info("Individual indexing completed successfully")

    try:
        if os.path.exists(f'./.byaldi/{index_name}'):
            RAG = RAGMultiModalModel.from_index(index_name, verbose=1)
            RAG.model.model = RAG.model.model.float()
            RAG.add_to_index(
                input_item=file_path,
                store_collection_with_index=True,
                doc_id=doc_id,
            )
        else:
            RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
            RAG.model.model = RAG.model.model.float()
            RAG.index(
                input_path=file_path,
                index_name=index_name,
                store_collection_with_index=True,
                overwrite=False,
                doc_ids=[doc_id],
            )
    except Exception as e:
        logger.exception(f"Error indexing PDF {filename}: {str(e)}")
        return jsonify({'error': 'Failed to index PDF', 'details': str(e)}), 500

    if file_hash in indexed_files:
        indexed_files[file_hash].append(index_name)
    else:
        indexed_files[file_hash] = [index_name]

    if index_name not in state["file_indices"]:
        state["file_indices"].append(index_name)

    save_state(state)  # Save state after updating indexed_files and file_indices

    os.remove(file_path)
    return jsonify({
        'message': 'File indexed successfully',
        'doc_id': doc_id,
        'title': title,
        'authors': authors
    })

def extract_text_metadata_with_qwen(text):
    # Initialize model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Split text into paragraphs
    paragraphs = text.split('\n\n')

    # Take approximately first 5 paragraphs or 2000 characters, whichever is shorter
    # This is roughly equivalent to first few pages worth of content
    preview_text = ''
    char_count = 0
    max_chars = 2000  # Adjust this based on model's context window

    for para in paragraphs[:5]:  # Look at first 5 paragraphs maximum
        if char_count + len(para) <= max_chars:
            preview_text += para + '\n\n'
            char_count += len(para) + 2
        else:
            # Add as much of the paragraph as possible without exceeding max_chars
            remaining_chars = max_chars - char_count
            preview_text += para[:remaining_chars] + '...'
            break

    # Prepare prompt with preview
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Given this excerpt from the beginning of a text: \n\n"
                    f"{preview_text}\n\n"
                    f"What is the title of this text? Who are the authors? "
                    f"Please respond in a json format with the keys Title and Authors. "
                    f"Authors should just be a list of names. "
                    f"If you cannot determine the title or authors, use empty values."
                )
            }
        ]
    }]

    try:
        # Prepare for inference
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate response
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,  # Lower temperature for more focused responses
            do_sample=False,  # Deterministic generation for metadata
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse response
        json_str = output_text
        if '```json' in json_str:
            json_str = json_str.split('```json\n')[1].split('\n```')[0]
        elif '```' in json_str:
            json_str = json_str.split('```\n')[1].split('\n```')[0]

        metadata = json.loads(json_str)
        title = metadata.get('Title', '').strip()
        authors = metadata.get('Authors', [])

        # Clean up authors
        if isinstance(authors, str):
            # Handle case where model returns authors as string
            authors = [author.strip() for author in authors.split(',')]
        if isinstance(authors, list):
            authors = ', '.join(author.strip() for author in authors if author.strip())

        # Validate results
        if not title or title.lower() in ['unknown', 'n/a', 'none']:
            title = 'Untitled Document'
        if not authors or authors.lower() in ['unknown', 'n/a', 'none']:
            authors = 'Unknown Author'

        return title, authors

    except Exception as e:
        logger.exception(f"Error extracting metadata from text: {str(e)}")
        return 'Untitled Document', 'Unknown Author'
    finally:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/indexText', methods=['POST'])
def index_text():
    logger.info("Starting index_text function")
    data = request.json

    if not data or 'text' not in data:
        logger.error("Missing text")
        return jsonify({'error': 'Missing text'}), 400

    text = data.get('text')
    index_name = data.get('index_name', 'universal')
    filename = data.get('filename', '')

    try:
        text_hash = calculate_text_hash(text)
        mapped = False
        doc_id = None
        title = None
        authors = None

        # Check if already mapped
        for current_doc_id, mapping in state["document_mapping"]["mappings"].items():
            if mapping['hash'] == text_hash:
                mapped = True
                doc_id = int(current_doc_id)
                title = mapping["title"]
                authors = mapping["authors"]
                break

        if not mapped:
            # Get next doc_id
            doc_id = state["document_mapping"]["next_id"]
            state["document_mapping"]["next_id"] += 1

            # Extract metadata
            try:
                title, authors = extract_text_metadata_with_qwen(text)
            except Exception as e:
                logger.error(f"Error extracting metadata: {str(e)}")
                title, authors = "Unknown", "Unknown"

            # Update state
            state["document_mapping"]["mappings"][doc_id] = {
                'hash': text_hash,
                'title': str(title),
                'authors': str(authors),
                'filename': filename
            }
            save_state(state)

        # Check if already indexed in this index
        if text_hash in state["indexed_texts"]:
            if index_name in state["indexed_texts"][text_hash]:
                return jsonify({'message': 'File already indexed', 'doc_id': doc_id})

        doc_id = str(doc_id)
        # Create individual index for the text
        try:
            logger.info(f"Creating new individual index for text {doc_id}")
            if not os.path.exists(f'./.ragatouille/{text_hash}'):
                RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", verbose=1)
                RAG.index(
                    index_name=text_hash,
                    collection=[text],
                    document_ids=[doc_id]
                )
        except Exception as e:
            logger.exception(f"Error creating individual index for text: {str(e)}")
            return jsonify({'message': 'Error creating individual index', 'doc_id': doc_id})

        # Add to main index
        try:
            if os.path.exists(f'./.ragatouille/{index_name}'):
                RAG = RAGPretrainedModel.from_index(index_name, verbose=1)
                RAG.add_to_index(
                    index_name=index_name,
                    new_collection=[text],
                    document_ids=[doc_id]
                )
            else:
                RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", verbose=1)
                RAG.index(
                    index_name=index_name,
                    collection=[text],
                    document_ids=[doc_id]
                )
        except Exception as e:
            logger.exception(f"Error adding text to main index: {str(e)}")
            return jsonify({'message': 'Error indexing to main index', 'doc_id': doc_id})

        # Update state with new index information
        if text_hash in state["indexed_texts"]:
            if index_name not in state["indexed_texts"][text_hash]:
                state["indexed_texts"][text_hash].append(index_name)
        else:
            state["indexed_texts"][text_hash] = [index_name]

        if index_name not in state["text_indices"]:
            state["text_indices"].append(index_name)

        save_state(state)

        return jsonify({
            'message': 'Texts indexed successfully',
            'index_name': index_name
        })

    except Exception as e:
        logger.exception("An error occurred during text indexing")
        return jsonify({"error": str(e)}), 500

def update_markdown_links(markdown_content, image_folder):
    pattern = r'!\[(.*?)\]\((.*?)\)'
    def replace_link(match):
        alt_text, image_path = match.groups()
        new_path = os.path.join(image_folder, os.path.basename(image_path))
        return f'![{alt_text}]({new_path})'
    return re.sub(pattern, replace_link, markdown_content)

@app.route('/convert', methods=['POST'])
def convert_pdf():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    if 'file' not in data or 'filename' not in data:
        return jsonify({"error": "Missing 'file' or 'filename' in request JSON"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            file_data = data['file']
            file_name = secure_filename(data['filename'])
            input_path = os.path.join(temp_dir, file_name)

            if file_data.startswith('data:application/pdf;base64,'):
                file_data = file_data.split(',')[1]

            try:
                pdf_content = base64.b64decode(file_data)
            except Exception as e:
                logger.error(f"Failed to decode base64 data: {e}")
                return jsonify({"error": "Invalid base64 data"}), 400

            with open(input_path, 'wb') as f:
                f.write(pdf_content)

            output_folder = os.path.join(temp_dir, 'output')
            os.makedirs(output_folder, exist_ok=True)

            command = f"marker_single {input_path} {output_folder} --batch_multiplier 2 --langs English"
            process = subprocess.run(command, shell=True, capture_output=True, text=True)

            if process.returncode == 0:
                images_folder = os.path.join(output_folder, os.path.splitext(file_name)[0], 'images')
                os.makedirs(images_folder, exist_ok=True)

                files_data = []

                for root, _, files in os.walk(output_folder):
                    for file in files:
                        if file.lower().endswith('.md'):
                            md_path = os.path.join(root, file)
                            with open(md_path, 'r', encoding='utf-8') as md_file:
                                content = md_file.read()
                            updated_content = update_markdown_links(content, 'images')
                            files_data.append({
                                "name": file,
                                "type": "markdown",
                                "content": updated_content
                            })

                for root, _, files in os.walk(output_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            img_path = os.path.join(root, file)
                            with open(img_path, 'rb') as img_file:
                                img_content = base64.b64encode(img_file.read()).decode('utf-8')
                            files_data.append({
                                "name": os.path.join('images', file),
                                "type": "image",
                                "content": img_content
                            })

                return jsonify({"success": True, "files": files_data})
            else:
                logger.error(f"Conversion failed: {process.stderr}")
                return jsonify({"error": "Conversion failed", "details": process.stderr}), 500
        except Exception as e:
            logger.exception("An error occurred during processing")
            return jsonify({"error": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if not file:
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Get the filename and check if it's compressed
        filename = file.filename
        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        filename = filename.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_input:
            file.save(temp_input.name)
            temp_input_path = temp_input.name

            try:
                # Handle compressed files
                if filename.endswith(('.zip', '.gz', '.bz2', '.tar', '.tar.gz', '.tar.bz2')):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_output:
                        if filename.endswith('.zip'):
                            import zipfile
                            with zipfile.ZipFile(temp_input_path, 'r') as zip_ref:
                                for member in zip_ref.namelist():
                                    if member.lower().endswith(('.mp3', '.wav', '.m4a', '.aac')):
                                        with zip_ref.open(member) as source:
                                            temp_output.write(source.read())
                                        break
                        # ... rest of the compression handling code ...

                        decompressed_file = temp_output.name
                else:
                    decompressed_file = temp_input_path

                # Process the audio file
                with open(decompressed_file, 'rb') as audio_file:
                    transcription = transcribe_audio_from_file(
                        audio_file,
                        model_size='large',
                        chunk_length=25,
                        stride_length=5,
                        temperature=0.2,
                        repetition_penalty=1.3
                    )

                return jsonify({"success": True, "transcription": transcription})

            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    if 'decompressed_file' in locals() and decompressed_file != temp_input_path:
                        os.unlink(decompressed_file)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary files: {e}")

    except Exception as e:
        logger.exception("An error occurred during transcription")
        return jsonify({"error": str(e)}), 500

# def transcribe_audio_from_file(file, model_size="large", chunk_length=30, stride_length=5, temperature=0.0, repetition_penalty=1.0):
#     try:
#         # Save the file content to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
#             # If file is a FileStorage object (from request.files)
#             if hasattr(file, 'save'):
#                 file.save(temp_file.name)
#             # If file is already a file object
#             else:
#                 temp_file.write(file.read())
#             temp_file_path = temp_file.name

#         # Load models
#         processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
#         model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")

#         # Move model to appropriate device
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = model.to(device)

#         # Load and process audio
#         audio, sr = librosa.load(temp_file_path, sr=16000)
#         chunk_size = int(chunk_length * sr)
#         stride_size = int(stride_length * sr)

#         transcription = ""
#         for i in range(0, len(audio), stride_size):
#             chunk = audio[i:i + chunk_size]
#             inputs = processor(
#                 chunk,
#                 return_tensors="pt",
#                 sampling_rate=16000
#             ).to(device)

#             with torch.no_grad():
#                 predicted_ids = model.generate(
#                     inputs.input_features,
#                     temperature=temperature,
#                     repetition_penalty=repetition_penalty,
#                     do_sample=temperature > 0.0,
#                 )

#             transcribed_text = processor.batch_decode(
#                 predicted_ids,
#                 skip_special_tokens=True
#             )[0]
#             transcription += transcribed_text + " "

#         return transcription.strip()

#     except Exception as e:
#         logger.exception("Error in transcribe_audio_from_file")
#         raise e

#     finally:
#         # Clean up temporary file
#         if 'temp_file_path' in locals():
#             try:
#                 os.unlink(temp_file_path)
#             except Exception as e:
#                 logger.error(f"Error deleting temporary file: {e}")

if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")
