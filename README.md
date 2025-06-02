# CH-SVV: A Benchmark for Multi-Type Chinese Harmful Short Video Detection
Core code, preview samples, and supplementary materials for the CH-SV dataset.
## 📁 Project Structure

We organize the repository into several parts, each corresponding to major components in our work:

### 🔧 `code/`
Contains the core implementation of **HAVE** (HArmful Video dEtection) :
- `code/models/StudentModel.py`: Main architecture of HAVE.
- `code/src/CrossmodalTransformer.py`: Implements **Cross-modal Semantic Interaction**.
- `code/src/STOG_different_modal.py`: Implements **Intra-modal Semantic Aggregation**.
- `code/video_prompting.py`: Implements **Video Semantic Prompting**.

### 📁 `supplementary_material/`
Includes supplementary materials associated with the paper:
- A demo video showcasing our **annotation system**.
- `.txt` files listing partial keywords derived from three **data collection strategies**:
  - **Normality_keywords.txt**: 10 domain-based keyword groups.
  - **Fakeness_keywords.txt**: 10 event-based keyword sets.
  - **Othertypes_keywords.txt** (Violence, Vulgarity, Offense, Danger): 10 keywords each.

### 🎞️ `video_samples/`
Provides a **preview subset** of the CH-SV dataset, with 10 sample videos for each of the six categories.

### 🧪 Test Set

We provide access to the **full test set videos** via Quark Cloud:

- 🔗 Download link: [https://pan.quark.cn/s/c5d2fa2cfad2](https://pan.quark.cn/s/c5d2fa2cfad2)  
- 🔐 Access code: `Zjmu`


