# Swift OCR: LLM Powered Fast OCR ⚡

## 🌟 Features

- **Flexible Input Options**: Accepts PDF files via direct upload or by specifying a URL.
- **Advanced OCR Processing**: Utilizes OpenAI's GPT-4 Turbo with Vision model for accurate text extraction.
- **Performance Optimizations**:
  - **Parallel PDF Conversion**: Converts PDF pages to images concurrently using multiprocessing.
  - **Batch Processing**: Processes multiple images in batches to maximize throughput.
  - **Retry Mechanism with Exponential Backoff**: Ensures resilience against transient failures and API rate limits.
- **Structured Output**: Extracted text is formatted using Markdown for readability and consistency.
- **Robust Error Handling**: Comprehensive logging and exception handling for reliable operations.
- **Scalable Architecture**: Asynchronous processing enables handling multiple requests efficiently.

## 📹 Demo

https://github.com/user-attachments/assets/6b39f3ea-248e-4c29-ac2e-b57de64d5d65

*Demo video showcasing the conversion of NASA's Apollo 17 flight documents, which include unorganized, horizontally and vertically oriented pages, into well-structured Markdown format without any issues.*

Here's a single, comprehensive section on cost comparison for your README:

## Cost Comparison and Value Proposition

Our solution offers an optimal balance of affordability, accuracy, and advanced features:

### Cost Breakdown
- Average token usage per image: ~1200
- Total tokens per page (including prompt): ~1500
- [GPT4O] Input token cost: $5 per million tokens
- [GPT4O] Output token cost: $15 per million tokens

For 1000 documents:
- Estimated total cost: $15

#### Cost Optimization Options
1. Utilizing GPT4 mini: Reduces cost to ~$8 per 1000 documents
2. Implementing batch API: Further reduces cost to ~$4 per 1000 documents

#### Market Comparison
This solution is significantly more affordable than alternatives:
- Our cost: $15 per 1000 documents
- CloudConvert: ~$30 per 1000 documents (PDFTron mode, 4 credits required)

While cost-effectiveness is a major advantage, our solution also provides:
- Superior accuracy and consistency
- Precise table generation
- Output in easily editable markdown format

This combination of affordability and advanced features makes solution stand out in the document processing market. It's not just about being cheaper; it's about providing excellent value through reliability, flexibility, and high-quality output.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)
- [Virtualenv](https://virtualenv.pypa.io/en/latest/) (optional but recommended)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/llm-openai-ocr.git
   cd llm-openai-ocr
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the root directory and add the following variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   OPENAI_DEPLOYMENT_ID=your_openai_deployment_id
   OPENAI_API_VERSION=your_openai_api_version  # Default is "gpt-4o"
   BATCH_SIZE=10  # Optional: Default is 1
   MAX_CONCURRENT_OCR_REQUESTS=5  # Optional: Default is 5
   MAX_CONCURRENT_PDF_CONVERSION=4  # Optional: Default is 4
   ```

   > **Note:** Replace `your_openai_api_key`, `your_azure_openai_endpoint`, and `your_openai_deployment_id` with your actual OpenAI credentials.

5. **Run the Application**

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.

## 🎯 Usage

### API Endpoint

**POST** `/ocr`

#### Request Parameters

- **file**: (Optional) Upload a PDF file.
- **ocr_request.url**: (Optional) URL of the PDF to process.

*You must provide either a file or a URL, not both.*

#### Example Using `curl`

**Uploading a PDF File:**

```bash
curl -X POST "http://127.0.0.1:8000/ocr" -F "file=@/path/to/your/document.pdf"
```

**Providing a PDF URL:**

```bash
curl -X POST "http://127.0.0.1:8000/ocr" -F "ocr_request={\"url\": \"https://example.com/document.pdf\"}" -H "Content-Type: application/json"
```

#### Response

- **200 OK**

  ```json
  {
    "text": "Extracted and formatted text from the PDF."
  }
  ```

- **Error Responses**

  - `400 Bad Request`: Invalid input parameters.
  - `422 Unprocessable Entity`: Validation errors.
  - `500 Internal Server Error`: Processing errors.

## 🧰 Configuration

All configurations are managed via environment variables. Ensure you have a `.env` file set up with the necessary variables as described in the [Installation](#installation) section.

### Key Configuration Variables

- **OPENAI_API_KEY**: Your OpenAI API key.
- **AZURE_OPENAI_ENDPOINT**: The endpoint for Azure OpenAI service.
- **OPENAI_DEPLOYMENT_ID**: Deployment ID for the OpenAI model.
- **OPENAI_API_VERSION**: API version for OpenAI (default: "gpt-4o").
- **BATCH_SIZE**: Number of images to process per OCR request (default: 1).
- **MAX_CONCURRENT_OCR_REQUESTS**: Maximum number of concurrent OCR requests (default: 5).
- **MAX_CONCURRENT_PDF_CONVERSION**: Maximum number of concurrent PDF page conversions (default: 4).
