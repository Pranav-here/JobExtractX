# JobExtractX: Extracting Structured Data from Job Postings
## AI-powered job information extraction

---

## 1. Overview

### Data Information
- Job posting text → Structured JSON format
- Standardized schema for job attributes
- Fields: experience level, employment status, work location, salary, benefits, job functions, required skills, certifications, education requirements, etc.
- Training dataset: 11,067 labeled job postings (split 90/10 for train/validation)
- Evaluation dataset: 1,058 labeled postings (1,000 used for final evaluation)
- Data format: {'source': job_posting_text, 'target': json_extraction}

### Models Evaluated
- **FLAN-T5 Large**: Full fine-tuning (783M parameters)
- **FLAN-T5 XL with LoRA**: Two variants (batch size 2 and 4) (2.85B parameters)
- **Mistral-7B with LoRA**: Instruction-tuned LLM with parameter-efficient fine-tuning (7.25B parameters)

### Experiment Environment
- GPU: RTX 6000 (24GB VRAM)
- Training optimizations:
  - 8-bit quantization
  - LoRA for parameter-efficient fine-tuning
  - BF16 precision for larger models
  - Gradient accumulation

---

## 2. Training Details

### Training Data Split
- Training set: 90% of 11,067 examples (~9,960 job postings)
- Validation set: 10% of 11,067 examples (~1,107 job postings)
- Used for model training and hyperparameter tuning

### FLAN-T5 Large (783M params)
- Full parameter fine-tuning
- Batch size: 1 with gradient accumulation (16 steps)
- Learning rate: 3e-5
- Max source length: 1536 tokens
- Epochs: 3
- Loss function: Sequence-to-sequence cross-entropy loss
  - Padding tokens masked with -100 to exclude from loss computation
  - Gradient clipping with max_norm=1.0
- Optimizer: Adafactor with FP32 precision
- Training runtime: 7.3 hours
- Training throughput: 1.13 samples/second

### FLAN-T5 XL with LoRA - Batch Size 2 (2.85B params)
- Batch size: 2 with gradient accumulation (8 steps)
- 8-bit quantization to reduce memory usage
- LoRA config: r=8, alpha=16, targeting q/v projection matrices
- Learning rate: 2e-4
- Loss function: Sequence-to-sequence cross-entropy loss
  - Special handling for partially masked inputs to ensure valid loss calculation
- Optimizer: Paged AdamW 8-bit with BF16 mixed precision
- BF16 mixed precision
- Training runtime: 22.2 hours
- Training throughput: 0.37 samples/second
- Evaluation loss: 0.161

### FLAN-T5 XL with LoRA - Batch Size 4 (2.85B params)
- Batch size: 4 with gradient accumulation (4 steps)
- 8-bit quantization to reduce memory usage
- LoRA config: r=8, alpha=16, targeting q/v projection matrices
- Learning rate: 2e-4
- Loss function: Sequence-to-sequence cross-entropy loss
  - Special handling for partially masked inputs to ensure valid loss calculation
- Optimizer: Paged AdamW 8-bit with BF16 mixed precision
- BF16 mixed precision
- Training runtime: 21.4 hours
- Training throughput: 0.39 samples/second
- Evaluation loss: 0.161


### Mistral-7B with LoRA (7.25B params)
- 8-bit quantization (memory efficiency)
- Batch size: 1 with gradient accumulation (16 steps)
- Chat template formatting for instruction following
- LoRA config: r=8, alpha=16, targeting query/value projections
- Learning rate: 2e-4
- Training runtime: 56.0 hours
- Training throughput: 0.15 samples/second

---

## 3. Model Evaluation

### Evaluation Dataset
- Separate evaluation dataset of 1,058 labeled job postings
- 1,000 examples for final evaluation
- Ensures fair comparison across all models

### Evaluation Methodology
- JSON parsing success rate
- Field-level metrics
  - String fields: presence rate, accuracy rate
  - List fields: precision, recall, F1 score
- Overall extraction quality (mean F1 score)

### FLAN-T5 Large Results
- Parse success rate: ~0.85
- Mean F1 score: ~0.72
- Evaluation loss: 0.163
- Strengths: Good balance of accuracy and recall, stable JSON output
- Weaknesses: Limited context handling compared to larger models

### FLAN-T5 XL with LoRA Results
- Batch 2 variant: Parse rate ~0.88, Mean F1 ~0.76, Eval loss: 0.161
- Batch 4 variant: Parse rate ~0.87, Mean F1 ~0.75, Eval loss: 0.161
- Strengths: Better field extraction accuracy, especially for complex fields
- Weaknesses: Occasional JSON formatting issues

### Mistral-7B with LoRA Results
- Parse success rate: ~0.90
- Mean F1 score: ~0.79
- Evaluation loss: 1.079 (higher due to different architecture)
- Strengths: Best at extraction quality, highest parse success rate
- Weaknesses: Slower inference time, more complex deployment

---

## 4. Model Comparison

### Performance Comparison
- Table showing key metrics across models:
  | Model | Parse Success | Mean F1 | Training Time | Training Throughput | Eval Loss |
  |-------|--------------|---------|---------------|---------------------|-----------|
  | FLAN-T5 Large (783M) | 0.85 | 0.72 | 7.3 hrs | 1.13 samples/sec | 0.163 |
  | FLAN-T5 XL (2.85B) (b=2) | 0.88 | 0.76 | 22.2 hrs | 0.37 samples/sec | 0.161 |
  | FLAN-T5 XL (2.85B) (b=4) | 0.87 | 0.75 | 21.4 hrs | 0.39 samples/sec | 0.161 |
  | Mistral-7B (7.25B) | 0.90 | 0.79 | 56.0 hrs | 0.15 samples/sec | 1.079 |

### Analysis
- Best performing model: Mistral-7B with LoRA
- Tradeoffs:
  - FLAN-T5 Large: Fastest training (7.3 hrs) but lowest accuracy
  - FLAN-T5 XL: Good balance of accuracy and training time (~22 hrs)
  - Mistral-7B: Highest accuracy but requires ~56 hrs of training
- Training time increases exponentially with model size
- FLAN-T5 XL batch size variants show minimal difference in final performance

---

## 5. Additional Technical Insights

### Training Stability
- Gradient norm values:
  - FLAN-T5 Large: 5.00
  - FLAN-T5 XL (batch 2): 1.53
  - FLAN-T5 XL (batch 4): 0.84
  - Mistral-7B: 0.46
- Lower gradient norms for larger models indicate more stable training

### Architectural Differences
- Model hidden dimensions:
  - FLAN-T5 Large: d_model = 1024, d_ff = 2816
  - FLAN-T5 XL: d_model = 2048, d_ff = 5120
  - Mistral-7B: hidden_size = 4096, intermediate_size = 14336

### Optimizer & Precision Settings
- FLAN-T5 Large: Adafactor optimizer (FP32)
- FLAN-T5 XL & Mistral: paged_adamw_8bit with BF16 precision
- Memory efficiency techniques allow training larger models on same hardware
- Evaluation throughput (samples/second):
  - FLAN-T5 Large: 2.95
  - FLAN-T5 XL (batch 2): 1.36
  - FLAN-T5 XL (batch 4): 1.42
  - Mistral-7B: 0.53

---

## 6. Real JSON Output Samples

### Example 1: Software Engineering Job
```json
{
  "experience_level": "Senior",
  "employment_status": ["Full-time"],
  "work_location": "Remote",
  "salary": {"min": "120000", "max": "160000", "period": "year", "currency": "USD"},
  "benefits": ["Health insurance", "401(k) matching", "Unlimited PTO"],
  "job_functions": ["Software Engineering", "Backend Development"],
  "required_skills": {
    "programming_languages": ["Python", "JavaScript"],
    "frameworks": ["Django", "React"],
    "databases": ["PostgreSQL"],
    "other": ["AWS", "Docker"]
  }
}
```

### Example 2: Data Science Position
```json
{
  "experience_level": "Mid-level",
  "employment_status": ["Full-time"],
  "work_location": "Hybrid",
  "required_skills": {
    "programming_languages": ["Python", "R"],
    "tools": ["Jupyter", "Pandas", "scikit-learn"],
    "other": ["Machine Learning", "SQL", "Data Visualization"]
  },
  "required_minimum_degree": "Bachelor's"
}
```

---

## 7. Challenges & Fixes

### JSON Formatting Issues
- Challenge: Models often generated malformed JSON
- Solution: Implemented robust JSON cleanup functions
  - Brace counting and balancing
  - Field normalization
  - String escaping fixes

### Long Context Processing
- Challenge: Job descriptions can be lengthy
- Solution: 
  - Increased max sequence length to 1536 tokens
  - Used gradient accumulation to handle larger contexts
  - Truncation strategies

### Memory Constraints
- Challenge: Fitting large models on available GPUs
- Solution:
  - 8-bit quantization
  - LoRA for parameter-efficient fine-tuning
  - Gradient checkpointing
  - Mixed precision training

---

## 8. Future Possibilities

### Model Improvements
- Ensemble approaches combining multiple models
- Distilled models for production deployment
- Custom JSON decoding head for improved output formatting

### Feature Expansion
- Multi-language job posting support
- Additional schema fields (company information, posting date, etc.)
- Confidence scores for extracted fields

### Applications
- Job matching algorithm integration
- Salary analysis and compensation trends
- Skills gap identification
- Market demand analytics

---

## Thank You!
Questions?