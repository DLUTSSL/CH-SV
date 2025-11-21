# CH-SV: A Benchmark for Multi-Type Chinese Harmful Short Video Detection
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

### 📊 Dataset and Features Availability

We are pleased to announce that the **original video data** and **extracted features** of the CH-SV dataset are now publicly available. You can access them via Quark Cloud using the links below. 

- **Extracted Features**  
  - 🔗 Download link: [https://pan.quark.cn/s/d54351288504](https://pan.quark.cn/s/d54351288504)  
  - 🔐 Access code: `Smv1` 

### ⚖️ Ethical Considerations and Privacy

We are fully aware of the potential sensitivities surrounding harmful content. All videos included in the CH-SV dataset were publicly available on the internet at the time of collection. To mitigate privacy risks:

- No personally identifiable information (PII) was deliberately collected or included.
- Content was selected and annotated solely for academic research purposes.
- Annotation was conducted by trained annotators following detailed ethical guidelines.

The dataset is intended **only for non-commercial research** on harmful content detection and should not be used for profiling or other misuse scenarios. Users must ensure responsible use and compliance with local regulations when using the dataset.

### 📄 Licensing and Access

- The **code** in this repository is released under the [MIT License](./LICENSE).
- The **test dataset** and **supplementary materials** (including videos and keyword files) are released under the  
  [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Summary of Terms:
- ✅ Free to share (with attribution)
- ❌ No commercial use
- ❌ No modifications or derivative works
If you wish to use the dataset in your research or publication, **please contact us in advance** to ensure appropriate use and to receive citation information once the paper is published.



