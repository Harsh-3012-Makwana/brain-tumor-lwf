# Brain Tumor MRI Dataset Audit Report

- **Total Scanned Images**: 7200
- **Corrupted / Unreadable Images**: 0
- **Training Split Total**: 5600
- **Testing Split Total**: 1600

## 1. Class Distribution

| Split | Glioma | Meningioma | Notumor | Pituitary | Total |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training** | 1400 | 1400 | 1400 | 1400 | 5600 |
| **Testing** | 400 | 400 | 400 | 400 | 1600 |
| **Overall** | 1800 | 1800 | 1800 | 1800 | 7200 |

## 2. Image Resolutions & Geometry

- **Min Resolution**: 150 x 167
- **Max Resolution**: 1375 x 1446
- **Mean Resolution**: 453.3 x 456.9
- **Aspect Ratio Range**: 0.6448 to 1.7857

## 3. Duplicate Analysis & Cross-Split Leakage

- **Exact Duplicates within Training**: 138 groups
- **Exact Duplicates within Testing**: 15 groups
- **Exact Duplicates ACROSS Train vs Test (Direct Leakage)**: 0 pairs/groups
- **Near-Duplicates ACROSS Train vs Test (pHash Hamming <= 4)**: 2022 pairs

### Top Near-Duplicate Cross-Split Pairs (Hamming <= 4)

| Train Image | Test Image | Train Class | Test Class | Hamming Distance |
|:---|:---|:---:|:---:|:---:|
| Training/glioma/Tr-gl_100.jpg | Testing/glioma/Te-gl_192.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1028.jpg | Testing/glioma/Te-gl_93.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_103.jpg | Testing/glioma/Te-gl_214.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_103.jpg | Testing/glioma/Te-gl_289.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_1030.jpg | Testing/glioma/Te-gl_71.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1046.jpg | Testing/meningioma/Te-aug-me_1.jpg | glioma | meningioma | 4 |
| Training/glioma/Tr-gl_1057.jpg | Testing/glioma/Te-gl_311.jpg | glioma | glioma | 0 |
| Training/glioma/Tr-gl_1072.jpg | Testing/glioma/Te-gl_286.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1085.jpg | Testing/glioma/Te-gl_168.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_109.jpg | Testing/glioma/Te-gl_269.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1098.jpg | Testing/glioma/Te-gl_348.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1154.jpg | Testing/glioma/Te-gl_393.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_1164.jpg | Testing/glioma/Te-gl_262.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1195.jpg | Testing/glioma/Te-gl_54.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1210.jpg | Testing/glioma/Te-gl_367.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_1219.jpg | Testing/glioma/Te-gl_122.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_1270.jpg | Testing/glioma/Te-gl_345.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1271.jpg | Testing/glioma/Te-gl_169.jpg | glioma | glioma | 0 |
| Training/glioma/Tr-gl_1278.jpg | Testing/glioma/Te-gl_69.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_1283.jpg | Testing/glioma/Te-gl_80.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1286.jpg | Testing/glioma/Te-gl_286.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1297.jpg | Testing/glioma/Te-gl_156.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1308.jpg | Testing/glioma/Te-gl_260.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1314.jpg | Testing/glioma/Te-gl_398.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1329.jpg | Testing/glioma/Te-gl_94.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1352.jpg | Testing/glioma/Te-gl_328.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1363.jpg | Testing/glioma/Te-gl_19.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1364.jpg | Testing/glioma/Te-gl_379.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1364.jpg | Testing/glioma/Te-gl_386.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_1367.jpg | Testing/glioma/Te-gl_179.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_137.jpg | Testing/glioma/Te-gl_339.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_1398.jpg | Testing/glioma/Te-gl_241.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_158.jpg | Testing/glioma/Te-gl_69.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_172.jpg | Testing/glioma/Te-gl_162.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_172.jpg | Testing/glioma/Te-gl_359.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_176.jpg | Testing/glioma/Te-gl_317.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_192.jpg | Testing/glioma/Te-gl_80.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_20.jpg | Testing/glioma/Te-gl_347.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_214.jpg | Testing/glioma/Te-gl_267.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_225.jpg | Testing/glioma/Te-gl_234.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_252.jpg | Testing/glioma/Te-gl_17.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_258.jpg | Testing/meningioma/Te-aug-me_74.jpg | glioma | meningioma | 4 |
| Training/glioma/Tr-gl_265.jpg | Testing/glioma/Te-gl_41.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_266.jpg | Testing/glioma/Te-gl_354.jpg | glioma | glioma | 2 |
| Training/glioma/Tr-gl_276.jpg | Testing/glioma/Te-gl_305.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_278.jpg | Testing/glioma/Te-gl_325.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_315.jpg | Testing/glioma/Te-gl_63.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_337.jpg | Testing/glioma/Te-gl_199.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_342.jpg | Testing/glioma/Te-gl_274.jpg | glioma | glioma | 4 |
| Training/glioma/Tr-gl_348.jpg | Testing/glioma/Te-gl_67.jpg | glioma | glioma | 4 |
| ... | *and 1972 more* | | | |

