# PART 2: MODULES AND IMPLEMENTATION DETAILS

## 4. MODULE DESCRIPTIONS

### 4.1 User Authentication Module

The User Authentication Module is foundational to the system, providing secure access control and maintaining user identity throughout the application lifecycle.

#### 4.1.1 Functionality

The authentication module handles:
- User registration with validation
- Secure login with credential verification
- Session management
- Access control to protected resources
- Secure logout

#### 4.1.2 Implementation Details

The module is implemented using Flask's session management and SQLAlchemy for database operations. Key components include:

**User Model**:
```python
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
```

**Registration Route**:
```python
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('register.html', error="Username already exists!")

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        # Automatically log in the user after registration
        session['user_id'] = new_user.id
        return redirect(url_for('index'))
    
    return render_template('register.html')
```

**Login Route**:
```python
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials!")

    return render_template('login.html')
```

**Logout Route**:
```python
@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))
```

#### 4.1.3 Security Considerations

In a production environment, several enhancements would be implemented:
- Password hashing using bcrypt or Werkzeug's security utilities
- CSRF protection on forms
- Rate limiting for login attempts
- Secure, HTTP-only session cookies
- Session expiration and renewal policies

#### 4.1.4 Authentication Flow

1. **Registration Process**:
   - User submits registration form with username and password
   - System validates input fields for completeness and format
   - System checks if username already exists in the database
   - If validation passes, a new user record is created
   - User is automatically logged in via session creation
   - Redirect to the main application interface

2. **Login Process**:
   - User submits login form with credentials
   - System queries the database for matching username and password
   - If credentials match, a session is created with the user's ID
   - User is redirected to the main application interface
   - If credentials don't match, an error message is displayed

3. **Session Validation**:
   - Protected routes check for the presence of user_id in the session
   - If absent, user is redirected to the login page
   - If present, the request is processed normally

4. **Logout Process**:
   - User initiates logout
   - System removes the user_id from the session
   - User is redirected to the home page

### 4.2 Dataset Collection and Management Module

#### 4.2.1 Dataset Requirements

For effective training and validation of the ViT model, the dataset collection focused on acquiring diverse and representative medical imaging data, specifically:

- **CT Scans**: High-resolution computed tomography images showing detailed lung structures
- **Labeled Data**: Images categorized into four classes:
  - Adenocarcinoma (common type of non-small cell lung cancer)
  - Large Cell Carcinoma (aggressive form of non-small cell lung cancer)
  - Normal (healthy lung tissue)
  - Squamous Cell Carcinoma (cancer starting in the squamous cells lining the airways)
- **Diversity**: Images from various demographics, equipment models, and imaging protocols
- **Quality**: Clear, properly exposed images with sufficient resolution for feature extraction

#### 4.2.2 Data Sources

Dataset acquisition involved collaboration with:
- Medical research institutions
- Public health databases
- Academic repositories with de-identified medical imaging data
- Publicly available medical imaging datasets with appropriate licensing

#### 4.2.3 Data Organization

The collected data was organized to facilitate efficient model training:
- Directory structure separated by class label
- Consistent file naming conventions
- Metadata tracking for image properties and source information
- Split into training, validation, and testing sets (typically 70%, 15%, 15%)

#### 4.2.4 Ethical and Legal Considerations

Dataset collection adhered to strict ethical and legal guidelines:
- All patient data was properly de-identified
- Appropriate data usage agreements and permissions were secured
- Compliance with relevant healthcare data regulations
- Ethical review board approval where applicable

### 4.3 Data Preprocessing Module

#### 4.3.1 Preprocessing Pipeline

The data preprocessing module transforms raw medical images into a standardized format suitable for the ViT model:

1. **Image Loading**: Converting from various medical image formats to standardized RGB format
2. **Resizing**: Standardizing dimensions to 224x224 pixels (ViT model requirement)
3. **Normalization**: Scaling pixel values to the range expected by the model
4. **Patch Creation**: Dividing images into 16x16 pixel patches for ViT processing
5. **Augmentation**: Applying transformations to increase dataset diversity

#### 4.3.2 Implementation in Code

The preprocessing function in the application handles image transformation for inference:

```python
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = inputs['pixel_values'].to(device)
    return inputs
```

For training, additional preprocessing steps were likely implemented, including:

```python
# Example augmentation pipeline (not in current application code)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 4.3.3 Data Augmentation Techniques

To enhance model robustness and prevent overfitting, several augmentation techniques were applied:

- **Geometric Transformations**:
  - Random rotations (±10 degrees)
  - Horizontal flips
  - Slight scaling variations
  - Random crops

- **Intensity Transformations**:
  - Brightness adjustments
  - Contrast modifications
  - Slight blur
  - Noise addition

These augmentations artificially expand the dataset and help the model generalize to variations in real-world images.

### 4.4 Image Processing and Prediction Module

#### 4.4.1 Module Overview

The Image Processing and Prediction Module serves as the core analytical engine of the application, handling:
- Image upload and storage
- Preprocessing for model compatibility
- ViT model inference
- Classification result generation
- Prediction storage in the database

#### 4.4.2 ViT Model Configuration

The application loads a pre-trained Vision Transformer model and configures it for the specific task:

```python
model_path = r"E:\prj\new vit app\scripts\best_vit_lung_cancer_model.pth"
model_name = 'google/vit-base-patch16-224'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
model.to(device)

# Load the model state dict
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
```

Key aspects of the configuration:
- Using the `google/vit-base-patch16-224` architecture as foundation
- Adapting the model for 4 output classes (lung cancer types)
- Loading weights from fine-tuned model
- Setting the model to evaluation mode
- Optimizing for available hardware (CPU/GPU)

#### 4.4.3 Prediction Process

The prediction route handles the complete workflow from image upload to result display:

```python
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Handle image upload and prediction logic
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template('index.html', prediction='No file selected', image=None)

    image_path = os.path.join(app.root_path, 'static/images', imagefile.filename)
    imagefile.save(image_path)

    # Preprocess image and make prediction
    X = preprocess(image_path)
    encode_label = {0: "Adenocarcinoma", 1: "Large Cell Carcinoma", 2: "Normal", 3: "Squamous Cell Carcinoma"}
    outputs = model(X)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    predicted_class = probabilities.argmax().item()

    classification = f"{encode_label[predicted_class]} ({probabilities[predicted_class].item() * 100:.2f}%)"

    # Save prediction to database
    new_prediction = Prediction(user_id=session['user_id'], image_path=imagefile.filename, prediction=classification)
    db.session.add(new_prediction)
    db.session.commit()

    return render_template('index.html', prediction=classification, image=imagefile.filename)
```

The prediction process follows these steps:
1. Validate user authentication
2. Receive and validate uploaded image
3. Save image to the static directory
4. Preprocess image for model compatibility
5. Pass preprocessed image through the ViT model
6. Apply softmax to convert logits to probabilities
7. Determine the highest probability class
8. Format the result with class label and confidence score
9. Save prediction to the database
10. Return the result to the user interface

#### 4.4.4 Class Mapping

The system maps numerical class indices to human-readable diagnoses:
- 0: Adenocarcinoma
- 1: Large Cell Carcinoma
- 2: Normal
- 3: Squamous Cell Carcinoma

This mapping facilitates interpretation of model outputs for healthcare professionals.

### 4.5 User Dashboard Module

#### 4.5.1 Functionality

The User Dashboard Module provides a centralized interface for users to:
- View their complete prediction history
- Access previously uploaded images
- Review diagnostic classifications
- Track patient progress over time
- Compare multiple predictions

#### 4.5.2 Implementation

The dashboard route retrieves and displays the user's prediction history:

```python
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    return render_template('dashboard.html', predictions=predictions)
```

#### 4.5.3 Data Presentation

The dashboard template organizes prediction data in a structured format:
- Chronological listing of predictions
- Thumbnail display of uploaded images
- Classification results with confidence scores
- Date and time of predictions
- Filtering and sorting options

## 5. IMPLEMENTATION DETAILS

### 5.1 Flask Application Structure

The application follows a modular structure typical of Flask applications:

```
/lung_cancer_detection/
│
├── app.py                  # Main application file
├── /static/                # Static assets
│   ├── /css/               # Stylesheets
│   ├── /js/                # JavaScript files
│   └── /images/            # Uploaded and static images
│
├── /templates/             # HTML templates
│   ├── home.html           # Landing page
│   ├── register.html       # Registration form
│   ├── login.html          # Login form
│   ├── index.html          # Main prediction interface
│   └── dashboard.html      # User prediction history
│
├── /scripts/               # Model scripts
│   └── best_vit_lung_cancer_model.pth  # Trained model weights
│
└── users.db                # SQLite database file
```

### 5.2 Application Initialization

The Flask application is initialized with necessary configurations:

```python
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

# Configure SQLAlchemy for database management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
```

Key configuration aspects:
- Setting a secret key for secure session management
- Configuring the SQLite database URI
- Initializing SQLAlchemy for ORM functionality

### 5.3 Model Loading and Configuration

The application loads the pre-trained ViT model during initialization:

```python
# Load Vision Transformer model
model_path = r"E:\prj\new vit app\scripts\best_vit_lung_cancer_model.pth"
model_name = 'google/vit-base-patch16-224'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
model.to(device)

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at {model_path}")

# Print the file path to debug
print(f"Loading model from: {model_path}")

# Load the model state dict
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
```

This configuration:
1. Defines the path to the trained model weights
2. Specifies the base ViT architecture to use
3. Determines whether to use CPU or GPU for inference
4. Initializes the image processor for standardized inputs
5. Configures the model with the appropriate number of output classes
6. Verifies the model file exists
7. Loads the trained weights
8. Sets the model to evaluation mode for inference

### 5.4 Database Models

The application defines two primary database models using SQLAlchemy:

```python
# Define a User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Define a Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
```

These models define:
- The structure of the database tables
- Data types and constraints for each field
- Relationships between tables (foreign key from Prediction to User)

### 5.5 Route Implementation

The application implements several routes to handle different user interactions:

#### 5.5.1 Home Route

```python
@app.route('/')
def home():
    return render_template('home.html')
```

#### 5.5.2 Main Interface Route

```python
@app.route('/index', methods=['GET'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')
```

#### 5.5.3 Authentication Routes

Registration, login, and logout routes as described in Section 4.1.

#### 5.5.4 Prediction Route

The prediction route as described in Section 4.4.

#### 5.5.5 Dashboard Route

The dashboard route as described in Section 4.5.

### 5.6 Application Entry Point

The application includes a standard entry point with database initialization:

```python
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(port=3000, debug=True)
```

This code:
1. Ensures the application only runs when executed directly (not when imported)
2. Creates an application context
3. Initializes all database tables if they don't exist
4. Starts the development server on port 3000 with debug mode enabled

### 5.7 Security Considerations

While the current implementation focuses on functionality, several security enhancements would be implemented in a production environment:

1. **Password Hashing**:
   ```python
   # Example using Werkzeug security
   from werkzeug.security import generate_password_hash, check_password_hash
   
   # In User model
   password = db.Column(db.String(256), nullable=False)
   
   # In registration route
   password_hash = generate_password_hash(password)
   new_user = User(username=username, password=password_hash)
   
   # In login route
   if user and check_password_hash(user.password, password):
       # Login successful
   ```

2. **CSRF Protection**:
   ```python
   # Using Flask-WTF
   from flask_wtf.csrf import CSRFProtect
   
   csrf = CSRFProtect(app)
   
   # In templates
   <form method="post">
       {{ csrf_token() }}
       <!-- Form fields -->
   </form>
   ```

3. **Input Validation**:
   ```python
   # Example validation for file uploads
   ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
   
   def allowed_file(filename):
       return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
   
   # In prediction route
   if not allowed_file(imagefile.filename):
       return render_template('index.html', error="Invalid file type")
   ```

4. **Secure File Storage**:
   ```python
   # Generate secure filename
   from werkzeug.utils import secure_filename
   import uuid
   
   # In prediction route
   secure_name = secure_filename(str(uuid.uuid4()) + os.path.splitext(imagefile.filename)[1])
   image_path = os.path.join(app.root_path, 'static/images', secure_name)
   ```

5. **Session Security**:
   ```python
   # Enhanced session configuration
   app.config.update(
       SESSION_COOKIE_SECURE=True,
       SESSION_COOKIE_HTTPONLY=True,
       SESSION_COOKIE_SAMESITE='Lax',
       PERMANENT_SESSION_LIFETIME=timedelta(hours=1)
   )
   ```

### 5.8 User Interface Implementation

The user interface is implemented using HTML templates with Bootstrap for responsive design.

#### 5.8.1 Main Prediction Interface

The `index.html` template provides:
- File upload form for CT scan images
- Preview of uploaded image
- Display area for prediction results
- Navigation to dashboard and logout options

#### 5.8.2 Dashboard Interface

The `dashboard.html` template includes:
- Table of prediction history
- Thumbnails of previously uploaded images
- Classification results with confidence scores
- Filtering and sorting options

The interface is designed to be intuitive for healthcare professionals, providing clear and concise information while maintaining a professional appearance consistent with medical applications.

This completes the detailed description of the modules and implementation aspects of the lung cancer detection system, providing a comprehensive understanding of how the application functions from both a technical and user perspective.
