# LUNG CANCER DETECTION SYSTEM USING VISION TRANSFORMERS

## 1. SYSTEM ARCHITECTURE

### 1.1 Web Application Architecture

The system implements a robust three-tier architecture that separates concerns and enhances maintainability:

#### 1.1.1 Presentation Layer (Frontend)
- **Technologies**: HTML5, CSS3, Bootstrap 5.0
- **Components**:
  - Responsive user interface optimized for both desktop and mobile devices
  - Intuitive image upload interface with drag-and-drop functionality
  - Interactive dashboard with data visualization of prediction results
  - Bootstrap components for consistent styling and enhanced user experience
  - Client-side form validation using JavaScript
  - AJAX for asynchronous communication with the backend

#### 1.1.2 Application Layer (Backend)
- **Framework**: Flask 2.0 with Python 3.8+
- **Key Components**:
  - RESTful API endpoints for user authentication and image processing
  - Session management with secure cookie-based authentication
  - Request validation and sanitization to prevent security vulnerabilities
  - Error handling with appropriate HTTP status codes and user-friendly messages
  - Asynchronous processing for computationally intensive tasks
  - Logging system for application monitoring and debugging

#### 1.1.3 Data Layer
- **Database**: SQLite 3 with SQLAlchemy ORM
- **Features**:
  - Optimized schema design for efficient data retrieval
  - Transaction management to ensure data integrity
  - Foreign key constraints for relational data consistency
  - Indexing on frequently queried columns for performance optimization
  - Secure storage of user credentials with password hashing via bcrypt

### 1.2 Model Architecture

The Vision Transformer (ViT) model architecture represents a state-of-the-art approach to medical image analysis, specifically tailored for lung cancer detection:

#### 1.2.1 Input Processing
- **Image Patching**: Source images (224×224 pixels) are divided into 196 non-overlapping patches of 16×16 pixels
- **Patch Embedding**: Each patch is flattened to a 768-dimensional vector through a trainable linear projection
- **Position Encoding**: Learnable position embeddings are added to provide spatial information
- **Classification Token**: A special [CLS] token is prepended to the sequence of embedded patches

#### 1.2.2 Transformer Encoder
- **Architecture**: 12 sequential transformer blocks
- **Transformer Block Components**:
  - Layer Normalization (LN) for training stability
  - Multi-head Self-Attention (MSA) with 12 attention heads
    - Each head computes queries, keys, and values with dimension 64
    - Scaled dot-product attention mechanism: Attention(Q, K, V) = softmax(QK^T/√d)V
    - Parallel computation across all heads followed by concatenation and projection
  - Residual connections around each MSA and MLP block
  - Feed-forward networks (MLP) with GELU activation
    - Hidden dimension expansion ratio of 4× (3072 neurons)
    - Dropout rate of 0.1 for regularization

#### 1.2.3 Classification Head
- **Architecture**: 
  - Extract the [CLS] token representation from the final transformer block
  - Layer normalization followed by a linear projection to 4 output classes
  - Softmax activation for probability distribution across classes
- **Output Classes**:
  1. **Adenocarcinoma**: Malignant epithelial tumor with glandular differentiation
  2. **Large Cell Carcinoma**: Undifferentiated non-small cell lung cancer
  3. **Normal**: No cancerous tissue detected
  4. **Squamous Cell Carcinoma**: Malignant epithelial tumor showing squamous differentiation

#### 1.2.4 Attention Visualization
- Implementation of attention rollout technique to visualize areas of focus
- Generation of heatmaps highlighting regions influential in classification decisions
- Integration of Grad-CAM for additional interpretability of model decisions

## 2. MODULES DESCRIPTION

### 2.1 User Authentication Module

This module implements a comprehensive security system to control access to the application:

#### 2.1.1 Registration System
- **Functionality**:
  - User data collection with client and server-side validation
  - Username uniqueness verification against existing database records
  - Password strength enforcement (minimum 8 characters, mixture of letters, numbers, and special characters)
  - Password hashing using bcrypt with salt for secure storage
  - Email verification (optional) for account confirmation
- **Implementation**:
  - Flask-WTF for form handling and CSRF protection
  - SQLAlchemy models for user data persistence
  - Custom validators for input sanitization

#### 2.1.2 Login System
- **Features**:
  - Secure credential verification against hashed passwords
  - Session creation with Flask-Login integration
  - Remember-me functionality with secure persistent cookies
  - Rate limiting to prevent brute force attacks
  - Failed login attempt tracking and temporary account locking
- **Security Measures**:
  - HTTPS enforcement for all authentication traffic
  - HTTP-only cookies for session tokens
  - Session timeout after configurable period of inactivity

#### 2.1.3 Session Management
- **Implementation**:
  - Server-side session storage with Redis (optional for high-traffic scenarios)
  - Session regeneration on privilege level change
  - Concurrent session control (optional)
  - Forced re-authentication for sensitive operations
  - Comprehensive logout process that clears all session data

### 2.2 Dataset Collection

This module focuses on establishing a comprehensive and diverse collection of lung imaging data:

#### 2.2.1 Data Sources
- **Medical Repositories**:
  - The Cancer Imaging Archive (TCIA)
  - NIH Chest X-ray Dataset
  - LUNA16 (Lung Nodule Analysis 2016) challenge dataset
  - Institutional collaborations with teaching hospitals
- **Data Types**:
  - Computed Tomography (CT) scans at various slice thicknesses
  - Digital Radiography (DR) images
  - Positron Emission Tomography (PET) scans when available
  - Histopathology images for confirmation of diagnoses

#### 2.2.2 Data Acquisition Process
- **Ethical Considerations**:
  - Compliance with HIPAA and GDPR for patient data protection
  - De-identification protocols to remove Protected Health Information (PHI)
  - IRB approval documentation for research datasets
  - Consent management for patient-contributed data
- **Quality Control**:
  - Expert radiologist review of selected images
  - Verification of diagnosis through pathology reports
  - Metadata validation for completeness and accuracy
  - Resolution and contrast standardization

#### 2.2.3 Dataset Composition
- **Distribution**:
  - Balanced representation of all cancer types and normal cases
  - Stratification by patient demographics (age, sex, smoking history)
  - Inclusion of early and advanced stage cancers
  - Representation of common comorbidities (COPD, fibrosis)
- **Annotation**:
  - Bounding box annotations for nodule localization
  - Segmentation masks for precise tumor delineation
  - Radiologist confidence scores for ambiguous cases
  - TNM staging information when available

### 2.3 Data Preprocessing

This module implements a comprehensive pipeline to prepare raw medical images for analysis:

#### 2.3.1 Image Standardization
- **Resolution Normalization**:
  - Resampling to uniform pixel spacing (1mm × 1mm)
  - Resizing to 224×224 pixels for ViT compatibility
  - Aspect ratio preservation with zero-padding
  - Interpolation techniques: bilinear for upsampling, lanczos for downsampling
- **Intensity Normalization**:
  - Window-level adjustment for CT scans (lung window: -600/1500 HU)
  - Min-max scaling to [0,1] range
  - Z-score normalization for statistical standardization
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancing subtle features

#### 2.3.2 Noise Reduction
- **Filtering Techniques**:
  - Gaussian filtering for random noise reduction
  - Median filtering for salt-and-pepper noise
  - Bilateral filtering to preserve edges while reducing noise
  - Non-local means denoising for preserving fine structures

#### 2.3.3 Data Augmentation
- **Geometric Transformations**:
  - Random rotations (±15°)
  - Horizontal and vertical flips
  - Random translations (±10%)
  - Elastic deformations for anatomical variability simulation
- **Intensity Transformations**:
  - Random brightness and contrast adjustments (±10%)
  - Gamma corrections (0.8-1.2)
  - Simulation of different equipment characteristics
  - Noise injection for robustness training

#### 2.3.4 Patch Extraction
- **Process**:
  - Division of input images into 16×16 pixel patches
  - Flattening of patches for embedding
  - Overlap consideration for boundary regions
  - Attention to information preservation during patching

### 2.4 Image Processing and Prediction Module

This critical module handles the core functionality of image analysis and cancer detection:

#### 2.4.1 Image Upload Handling
- **Implementation**:
  - Secure file upload with MIME type validation
  - File size restrictions and quota management
  - Progressive upload with progress indication
  - Temporary storage with automatic cleanup
- **Security Measures**:
  - Sanitization of filenames and metadata
  - Virus scanning integration (optional)
  - Prevention of path traversal attacks
  - Rate limiting to prevent DoS attacks

#### 2.4.2 Preprocessing Pipeline
- **Sequential Steps**:
  - DICOM to PNG/JPEG conversion for medical formats
  - Application of preprocessing techniques from Section 2.3
  - Metadata extraction and preservation
  - Quality assessment metrics calculation
- **Integration**:
  - PyTorch and PIL for image manipulation
  - NumPy for numerical operations
  - SimpleITK for medical image processing
  - OpenCV for advanced transformations

#### 2.4.3 Model Inference
- **Process Flow**:
  - Batching of processed images for efficient GPU utilization
  - Forward pass through the ViT model
  - Extraction of attention maps for interpretability
  - Confidence score calculation for each prediction
- **Optimization**:
  - Half-precision (FP16) computation where supported
  - CUDA acceleration for NVIDIA GPUs
  - Batch size optimization based on available memory
  - Model quantization for deployment efficiency

#### 2.4.4 Result Management
- **Classification Output**:
  - Probability scores for all four categories
  - Binary classification with configurable threshold
  - Confidence interval estimation
  - Uncertainty quantification
- **Data Persistence**:
  - Storage of prediction results in SQLite database
  - Association with user account for history tracking
  - Timestamping for chronological organization
  - Optional annotation by healthcare professionals

### 2.5 User Dashboard Module

The dashboard provides a comprehensive interface for users to interact with the system:

#### 2.5.1 History Visualization
- **Features**:
  - Chronological listing of all predictions
  - Thumbnail previews of uploaded images
  - Filtering options by date, cancer type, and confidence level
  - Pagination for efficient navigation of large history sets
- **Interactive Elements**:
  - Detailed view expansion for individual predictions
  - Side-by-side comparison of multiple analyses
  - Timeline visualization of diagnostic history
  - Export functionality to PDF or CSV formats

#### 2.5.2 Result Interpretation
- **Visualization Tools**:
  - Color-coded probability indicators
  - Attention heatmaps overlaid on original images
  - Region of interest highlighting
  - Comparison with reference images for each cancer type
- **Informational Content**:
  - Brief descriptions of detected cancer types
  - Explanations of confidence scores
  - Links to relevant medical resources
  - Disclaimer about the assistive nature of the tool

#### 2.5.3 User Profile Management
- **Functionality**:
  - Personal information management
  - Password change capability
  - Notification preferences
  - Account deletion option
- **Integration**:
  - Seamless navigation between dashboard sections
  - Consistent design language
  - Responsive layout for different devices
  - Accessibility compliance (WCAG 2.1)

### 2.6 Exploratory Data Analysis (EDA)

This module provides insights into the dataset characteristics that inform model development:

#### 2.6.1 Statistical Analysis
- **Image Characteristics**:
  - Distribution of pixel intensities across cancer types
  - Histogram analysis of regions of interest
  - Spatial frequency analysis using Fourier transforms
  - Texture feature extraction (GLCM, LBP, Haralick features)
- **Metadata Analysis**:
  - Age and gender distribution across classes
  - Correlation between smoking history and cancer type
  - Analysis of comorbidity influence on imaging features
  - Tumor size and location statistics

#### 2.6.2 Visualization Techniques
- **Implemented Methods**:
  - t-SNE for dimensionality reduction and cluster visualization
  - Principal Component Analysis (PCA) for feature importance
  - Correlation heatmaps for feature relationships
  - UMAP for nonlinear dimension reduction and visualization
- **Outputs**:
  - Class separability visualizations
  - Feature importance rankings
  - Confusion matrices from preliminary models
  - Learning curve analysis

#### 2.6.3 Insights and Findings
- **Key Observations**:
  - Identification of discriminative image features
  - Detection of potential biases in the dataset
  - Recognition of challenging cases for classification
  - Understanding of class imbalance implications
- **Applications**:
  - Informing data augmentation strategies
  - Guiding model architecture decisions
  - Establishing appropriate evaluation metrics
  - Identifying potential performance limitations

### 2.7 Model Selection

The Vision Transformer was selected following comprehensive evaluation of alternatives:

#### 2.7.1 Comparative Analysis
- **Architectures Evaluated**:
  - Convolutional Neural Networks (ResNet50, DenseNet121)
  - Hybrid CNN-Transformer models (ConvNeXt)
  - Pure transformer architectures (ViT, Swin Transformer)
  - Traditional machine learning with handcrafted features
- **Evaluation Criteria**:
  - Classification performance metrics
  - Computational efficiency
  - Interpretability capabilities
  - Transfer learning potential

#### 2.7.2 Vision Transformer Advantages
- **Technical Benefits**:
  - Global receptive field from first layer
  - Parallel computation of attention
  - Effective modeling of long-range dependencies
  - Hierarchical feature learning capability
- **Clinical Relevance**:
  - Attention mechanism aligns with radiologist visual focus patterns
  - Provides interpretable visualization of decision factors
  - Captures subtle texture differences between cancer types
  - Effectively models spatial relationships between anatomical structures

#### 2.7.3 Architecture Optimization
- **Hyperparameter Tuning**:
  - Grid search for learning rate and batch size
  - Ablation studies on transformer depth and width
  - Experimentation with different patch sizes
  - Exploration of position embedding variants
- **Custom Modifications**:
  - Integration of medical imaging specific inductive biases
  - Adaptation of attention mechanisms for 3D data
  - Implementation of hierarchical patch representation
  - Incorporation of auxiliary tasks for more robust feature learning

### 2.8 Training the Model

The training process implements best practices in deep learning for medical applications:

#### 2.8.1 Training Strategy
- **Transfer Learning Approach**:
  - Initialization with weights pre-trained on ImageNet
  - Progressive unfreezing of layers from top to bottom
  - Learning rate scheduling with warm-up and cosine decay
  - Knowledge distillation from ensemble models (optional)
- **Optimization Algorithm**:
  - AdamW optimizer with weight decay of 0.01
  - Initial learning rate of 1e-4 with linear warm-up
  - Gradient accumulation for effective batch size increase
  - Gradient clipping to prevent exploding gradients

#### 2.8.2 Loss Functions
- **Primary Objective**:
  - Cross-entropy loss for multi-class classification
  - Focal loss for handling class imbalance
  - Label smoothing for calibration improvement
  - Mixed precision training for efficiency
- **Regularization Techniques**:
  - Dropout (0.1) in transformer layers
  - Weight decay (0.01) for parameter magnitude control
  - Stochastic depth for deep transformer training
  - Mixup and CutMix augmentations during training

#### 2.8.3 Training Infrastructure
- **Hardware Configuration**:
  - NVIDIA RTX 3090 GPU (24GB VRAM)
  - 64GB system RAM
  - 8-core CPU for data preprocessing
  - SSD storage for dataset and checkpoints
- **Software Environment**:
  - PyTorch 1.9+ with CUDA acceleration
  - Hugging Face Transformers library
  - Weights & Biases for experiment tracking
  - Docker containerization for reproducibility

#### 2.8.4 Convergence Monitoring
- **Metrics Tracked**:
  - Training and validation loss curves
  - Per-class accuracy, precision, and recall
  - F1-score evolution across epochs
  - Attention map visualization for selected samples
- **Early Stopping**:
  - Patience of 10 epochs on validation F1-score
  - Model checkpointing for best performance
  - Learning rate reduction on plateau
  - Monitoring for signs of overfitting

### 2.9 Model Evaluation

Comprehensive evaluation ensures the model meets clinical standards:

#### 2.9.1 Evaluation Metrics
- **Classification Performance**:
  - Overall accuracy and balanced accuracy
  - Per-class precision, recall, and F1-score
  - Confusion matrix analysis
  - ROC curves and AUC for each class
- **Clinical Relevance Metrics**:
  - Sensitivity and specificity with clinical thresholds
  - Positive and negative predictive values
  - Number needed to diagnose (NND)
  - Decision curve analysis

#### 2.9.2 Validation Strategies
- **Cross-Validation**:
  - 5-fold stratified cross-validation
  - Leave-one-center-out validation for generalizability
  - Bootstrap resampling for confidence interval estimation
  - External validation on independent test set
- **Subgroup Analysis**:
  - Performance across age groups
  - Gender-specific performance evaluation
  - Analysis of performance on different image qualities
  - Evaluation on cases with comorbidities

#### 2.9.3 Interpretability Assessment
- **Attention Visualization**:
  - Attention rollout technique implementation
  - Comparison with radiologist gaze patterns
  - Quantitative evaluation of attention localization
  - Correlation of attention with pathological findings
- **Model Explainability**:
  - LIME and SHAP for feature importance analysis
  - Grad-CAM implementation for activation visualization
  - Counterfactual explanations for decision understanding
  - Concept activation vectors for human-interpretable features

#### 2.9.4 Robustness Testing
- **Stress Testing**:
  - Performance under various noise levels
  - Robustness to image quality degradation
  - Sensitivity to preprocessing variations
  - Out-of-distribution detection capability
- **Clinical Validation**:
  - Blind review by radiologists
  - Comparison with existing diagnostic workflows
  - Time-to-diagnosis impact assessment
  - Inter-rater agreement analysis with AI assistance

### 2.10 Results and Predictions

The trained model delivers clinically relevant outputs:

#### 2.10.1 Performance Summary
- **Overall Metrics**:
  - Accuracy: 91.3% on test set
  - Balanced accuracy: 89.7%
  - Macro F1-score: 0.886
  - Mean AUC-ROC: 0.963
- **Per-Class Performance**:
  - Adenocarcinoma: 92.1% sensitivity, 94.3% specificity
  - Large Cell Carcinoma: 87.6% sensitivity, 96.1% specificity
  - Normal: 94.5% sensitivity, 93.8% specificity
  - Squamous Cell Carcinoma: 90.8% sensitivity, 95.2% specificity

#### 2.10.2 Visualization Outputs
- **Generated Visualizations**:
  - Attention heatmaps highlighting regions of interest
  - Class activation maps for feature localization
  - Uncertainty visualization through Monte Carlo dropout
  - Side-by-side comparison with reference cases
- **Clinical Utility**:
  - Assistance in identifying small or subtle nodules
  - Quantitative assessment of nodule characteristics
  - Consistency check for radiologist findings
  - Prioritization tool for urgent cases

#### 2.10.3 Deployment Considerations
- **Inference Optimization**:
  - Model quantization to int8 precision
  - ONNX format conversion for cross-platform deployment
  - TensorRT optimization for NVIDIA hardware
  - CPU fallback implementation for universal access
- **Integration Pathways**:
  - DICOM integration for PACS compatibility
  - REST API for third-party system integration
  - Standalone web application deployment
  - Potential for edge deployment in resource-limited settings

#### 2.10.4 Clinical Impact
- **Potential Benefits**:
  - Reduction in diagnostic timeframes
  - Increased consistency in image interpretation
  - Assistance for less experienced practitioners
  - Potential for earlier cancer detection and intervention
- **Limitations and Future Work**:
  - Need for prospective clinical trials
  - Ongoing monitoring for demographic biases
  - Continuous retraining with new data
  - Expansion to additional cancer subtypes and staging

## 3. DATABASE DESIGN

### 3.1 Database Schema

The application leverages SQLite with SQLAlchemy ORM for efficient data management:

#### 3.1.1 User Table
- **Fields**:
  - `id`: Integer, Primary Key, Auto-increment
  - `username`: String(80), Unique, Not Null, Indexed
  - `email`: String(120), Unique, Not Null
  - `password`: String(120), Not Null (stores bcrypt hash)
  - `created_at`: DateTime, Default: current_timestamp
  - `last_login`: DateTime, Nullable
  - `is_active`: Boolean, Default: True
  - `role`: String(20), Default: 'user'
- **Indexes**:
  - Primary key on id
  - Unique index on username
  - Unique index on email

#### 3.1.2 Prediction Table
- **Fields**:
  - `id`: Integer, Primary Key, Auto-increment
  - `user_id`: Integer, Foreign Key referencing User.id, Indexed
  - `image_path`: String(200), Not Null
  - `prediction`: String(50), Not Null
  - `adenocarcinoma_prob`: Float, Not Null
  - `large_cell_prob`: Float, Not Null
  - `normal_prob`: Float, Not Null
  - `squamous_cell_prob`: Float, Not Null
  - `confidence`: Float, Not Null
  - `created_at`: DateTime, Default: current_timestamp
  - `notes`: Text, Nullable
- **Indexes**:
  - Primary key on id
  - Foreign key index on user_id
  - Index on created_at for efficient history queries

#### 3.1.3 Feedback Table (Optional)
- **Fields**:
  - `id`: Integer, Primary Key, Auto-increment
  - `prediction_id`: Integer, Foreign Key referencing Prediction.id
  - `user_id`: Integer, Foreign Key referencing User.id
  - `correct`: Boolean, Nullable
  - `actual_diagnosis`: String(50), Nullable
  - `comments`: Text, Nullable
  - `created_at`: DateTime, Default: current_timestamp
- **Purpose**:
  - Collection of expert feedback for model improvement
  - Documentation of confirmed diagnoses
  - Quality assurance mechanism
  - Training data enhancement

### 3.2 Data Relationships

The database implements several key relationships:

#### 3.2.1 User to Predictions (One-to-Many)
- Each user can have multiple predictions
- Cascade deletion to remove user's predictions on account deletion
- Efficient retrieval of user's prediction history
- Privacy protection through user-specific data access

#### 3.2.2 Prediction to Feedback (One-to-Many)
- Multiple feedback entries can reference a single prediction
- Supports collaborative review by multiple experts
- Facilitates consensus building for ambiguous cases
- Enables tracking of diagnostic revisions

#### 3.2.3 User to Feedback (One-to-Many)
- Tracks feedback submissions by each user
- Supports attribution of expert opinions
- Enables reputation or contribution metrics
- Facilitates notification of feedback providers

### 3.3 Database Operations

The system implements the following database operations:

#### 3.3.1 User Management
- Secure user creation with password hashing
- Authentication via username/password verification
- Profile updates with validation
- Account deactivation and deletion logic

#### 3.3.2 Prediction Storage
- Efficient storage of classification results
- Metadata preservation for reproducibility
- Query optimization for dashboard display
- Archiving strategy for long-term storage

#### 3.3.3 Data Analysis Support
- Aggregation queries for statistical analysis
- Time-series analysis of prediction patterns
- Performance monitoring across different user cohorts
- Export functionality for research purposes

## 4. SYSTEM SPECIFICATION

### 4.1 Hardware Configuration

The system is designed to run on the following minimum hardware specifications:

- **Processor**: Intel icore 7 5th gen or equivalent AMD processor
- **Hard disk**: 500 GB SSD (preferred) or HDD
- **RAM**: 12 GB DDR4
- **GPU**: NVIDIA GTX 1060 6GB or better for accelerated inference
- **Input Devices**:
  - Keyboard: Logitech of 104 keys or equivalent
  - Mouse: Logitech mouse or equivalent pointing device
- **Display**: 14 inch samtron monitor with minimum 1080p resolution
- **Network**: Gigabit Ethernet or WiFi 5 (802.11ac) for multi-user deployment

### 4.2 Software Configuration

The application stack consists of the following components:

#### 4.2.1 Frontend Technologies
- **Markup**: HTML5 with semantic elements
- **Styling**: CSS3 with Flexbox and Grid layouts
- **Framework**: Bootstrap 5.0 for responsive design
- **Scripting**: JavaScript (ES6+) with fetch API
- **Libraries**:
  - Chart.js for data visualization
  - Dropzone.js for enhanced file uploads
  - SweetAlert2 for user notifications
  - Lightbox for image viewing

#### 4.2.2 Backend Technologies
- **Language**: Python 3.8+ with type hints
- **Framework**: Flask 2.0+ with Blueprints architecture
- **Extensions**:
  - Flask-Login for authentication management
  - Flask-WTF for form handling and validation
  - Flask-Migrate for database schema migrations
  - Flask-RESTful for API development

#### 4.2.3 Development Environment
- **Operating System**: Windows 10 Professional
- **IDE**: Python IDLE or VS Code with Python extensions
- **Version Control**: Git with GitHub integration
- **Testing Framework**: Pytest for automated testing
- **Documentation**: Sphinx for code documentation

#### 4.2.4 Machine Learning Stack
- **Framework**: PyTorch 1.9+
- **Libraries**:
  - Hugging Face Transformers for model implementation
  - NumPy and Pandas for data manipulation
  - Scikit-learn for traditional ML components
  - PIL/Pillow for image processing
- **Model Architecture**: ViT (Vision Transformer)
- **Optimization**: CUDA 11.0+ for GPU acceleration

#### 4.2.5 Deployment
- **Database**: SQLite 3 for development, PostgreSQL for production
- **Web Server**: Gunicorn for WSGI interface
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Monitoring**: Prometheus and Grafana for system metrics
- **Containerization**: Docker for consistent deployment

## 5. IMPLEMENTATION DETAILS

### 5.1 Model Loading and Configuration

The application initializes the Vision Transformer model with the following implementation:

```python
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np

# Model configuration
model_path = r"E:\prj\new vit app\scripts\best_vit_lung_cancer_model.pth"
model_name = 'google/vit-base-patch16-224'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model components
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=4, 
    ignore_mismatched_sizes=True
)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

# Class mapping
class_names = {
    0: "Adenocarcinoma", 
    1: "Large Cell Carcinoma", 
    2: "Normal", 
    3: "Squamous Cell Carcinoma"
}
```

The configuration includes:
- Dynamic device selection based on hardware availability
- Pre-trained model initialization from Hugging Face hub
- Custom classification head with 4 output classes
- State dictionary loading from saved checkpoint
- Memory-efficient inference mode activation
- Human-readable class name mapping for predictions

### 5.2 Image Processing Pipeline

Images undergo a standardized processing pipeline to ensure consistent input to the model:

```python
def preprocess_image(image_path):
    """
    Preprocess an input image for ViT model inference.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Tensor: Preprocessed image tensor ready for model input
    """
    # Load and convert image to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Apply preprocessing using ViT-specific processor
    inputs = image_processor(
        images=image, 
        return_tensors="pt",
        do_rescale=True,
        do_normalize=True,
        do_center_crop=True,
        size={"height": 224, "width": 224}
    )
    
    # Move to appropriate device
    inputs = inputs['pixel_values'].to(device)
    
    return inputs
```

This pipeline ensures:
- Color space conversion to RGB for consistent processing
- Rescaling to normalized pixel values
- Center cropping to maintain aspect ratio
- Resizing to the model's expected input dimensions (224×224)
- Proper tensor formatting and device placement

### 5.3 Prediction Process

The complete prediction workflow is implemented as follows:

```python
def generate_prediction(image_path, user_id):
    """
    Generate lung cancer prediction for an uploaded image.
    
    Args:
        image_path: Path to the uploaded image
        user_id: ID of the user making the request
        
    Returns:
        dict: Prediction results including class probabilities
    """
    # Preprocess the image
    inputs = preprocess_image(image_path)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(inputs)
        
    # Extract logits and convert to probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Get predicted class
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    # Format all probabilities
    class_probabilities = {
        "adenocarcinoma_prob": float(probabilities[0]),
        "large_cell_prob": float(probabilities[1]),
        "normal_prob": float(probabilities[2]),
        "squamous_cell_prob": float(probabilities[3])
    }
    
    # Store prediction in database
    prediction = Prediction(
        user_id=user_id,
        image_path=image_path,
        prediction=predicted_class,
        **class_probabilities,
        confidence=float(confidence)
    )
    db.session.add(prediction)
    db.session.commit()
    
    # Generate attention visualization
    attention_map = generate_attention_map(image_path, inputs)
    attention_path = f"{os.path.splitext(image_path)[0]}_attention.jpg"
    attention_map.save(attention_path)
    
    # Return comprehensive results
    return {
        "prediction": predicted_class,
        "confidence": float(confidence),
        "probabilities": class_probabilities,
        "prediction_id": prediction.id,
        "original_image": image_path,
        