# 🩻 EfficientNet-PUV-Classifier 🧬

This repository provides the official implementation of a deep learning pipeline 
for the automatic classification of **Posterior Urethral Valves** (**PUV**) from **Voiding CystoUrethroGram** (**VCUG**) images, 
based on **EfficientNet** **architectures**.

**Posterior Urethral Valves** (**PUV**) are a rare congenital anomaly affecting the male urethra. 
Early and accurate diagnosis is crucial to prevent serious complications, including renal damage. 
This project aims to **support the diagnostic process** by exploiting _Convolutional Neural Networks_ (_CNN_) to identify PUV cases directly from imaging data.

## 🌍 Impact and Relevance
- Contributes to the **early diagnosis of PUV**, which is essential to avoid long-term renal complications.
- Provides a **reproducible framework** that can be adapted or extended to other urological conditions.
- Promotes **collaboration** between **clinicians** and **AI researchers**, bridging the gap between computer vision and healthcare.

## 🗂️ Dataset
The dataset used in this study originates from a real-world acquisition campaign conducted within the activity of the [SUD4VUP project](https://sites.google.com/view/sud4vup/home), coordinated by the University of Campania “Luigi Vanvitelli”. <br>
The data collection involved the Nephrology and Urology Units at ["Luigi Vanvitelli" University Hospital](https://www.policliniconapoli.it/home), [AORN Santobono-Pausilipon](https://www.santobonopausilipon.it) in Naples, and [IRCCS Ca’ Granda](https://www.policlinico.mi.it) in Milan. <br>
The dataset includes routine **VCUG acquisitions** from **male pediatric patients**, each labeled to indicate the presence or absence of **Posterior Urethral Valves** (**PUV**).

### 🔐 Data Availability
Due to privacy regulations and clinical data protection protocols, **the dataset is not publicly available**. 
However, preprocessing scripts and data handling instructions are included to support reproducibility using compatible datasets.

Researchers interested in collaboration or data access for institutional research purposes may contact the **project coordinators**.

## 👤 Project Team
The development of this repository involved contributions from a multidisciplinary team of researchers, including:
- [**Ciro Russo**](https://www.linkedin.com/in/ciro-russo-phd-b14056100/) - PostDoctoral Researcher, University of Cassino and Lazio Meridionale
- [**Gaetano Settembre**](https://www.linkedin.com/in/gaetano-settembre/) - PhD Student in Computer Science and Mathematics, University of Bari Aldo Moro
- [**Grazia Gargano**](https://www.linkedin.com/in/grazia-gargano-307124189/) - PhD Student in Computer Science and Mathematics, University of Bari Aldo Moro
- [**Maria Stella de Biase**](https://www.linkedin.com/in/maria-stella-de-biase-711ba0171/) - Researcher, University of Campania "Luigi Vanvitelli"
- [**Roberta De Fazio**](https://www.linkedin.com/in/roberta-de-fazio-phd-1836b2226/) - PostDoctoral Fellowship, University of Campania "Luigi Vanvitelli"

## 🤝🏼 Acknowledgements

This work was developed through a collaboration between:

- [University of Cassino and Lazio Meridionale](https://www.unicas.it)
- [University of Bari "Aldo Moro"](https://www.uniba.it/it)
- [University of Campania "Luigi Vanvitelli"](https://www.unicampania.it) (coordinator of the **SUP4VUP project** and data provider)


## 📝 Citation
If you find the project codes useful for your research, please consider citing

```
@InProceedings{PUVClassificationICIAP25,
  title = {{AI in Pediatric Urology: Deep Learning-based Approach supporting Posterior Urethral Valves Diagnosis on VCUG Imaging}},
  ISBN = {9783032113818},
  ISSN = {1611-3349},
  editor = {Rodotà, Emanuele and Galasso, Fabio and Masi, Iacopo},
  DOI = {10.1007/978-3-032-11381-8_12},
  booktitle = {Image Analysis and Processing - ICIAP 2025 Workshops},
  publisher = {Springer Nature Switzerland},
  address = {Cham},
  author = {Russo,  Ciro and Settembre,  Gaetano and Gargano,  Grazia and {de Biase},  Maria Stella and {De Fazio},  Roberta},
  year = {2026},
  pages = {138–149}
}
```
