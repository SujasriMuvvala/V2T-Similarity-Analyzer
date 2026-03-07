# V2T-Similarity-Analyzer
A multimodal AI system that analyzes a video and determines whether its content matches a given reference answer.
The system extracts audio and visual information, converts them into natural language descriptions, and evaluates semantic similarity between the generated description and a reference answer.A multimodal AI system that analyzes a video and determines whether its content matches a given reference answer.
The system extracts audio and visual information, converts them into natural language descriptions, and evaluates semantic similarity between the generated description and a reference answer.

🚀 Features
Upload a video through a web interface
Extract audio from the video
Convert speech to text using Whisper
Extract frames from the video using OpenCV
Generate scene descriptions for frames using caption generation models
Convert sentences into semantic embeddings using Sentence Transformers
Compute similarity between generated captions and reference answer using cosine similarity

Video Upload
      ↓
Audio Extraction
      ↓ 
Speech → Text (Whisper) 
      ↓
Frame Extraction (OpenCV) 
      ↓
Caption Generation 
      ↓
Semantic Embeddings (SentenceTransformer)
      ↓
Cosine Similarity Calculation
      ↓
Best Matching Caption + Similarity Score

⚙️ Technologies Used
Python
Flask – backend web framework
OpenCV – video frame processing
Whisper – speech-to-text transcription
Sentence Transformers – semantic embeddings
scikit-learn – cosine similarity calculatio

🎯 Use Cases
Automated video evaluation
Educational video assessment
Interview analysis
Media content indexing
Semantic video search
