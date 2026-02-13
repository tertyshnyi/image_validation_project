# Prehľad modelov a knižníc s ohľadom na licenciu

---

# 1. Modely na detekciu objektov

## YOLOv8 (Ultralytics)

Vyvinuté: Ultralytics  
Repository: https://github.com/ultralytics/ultralytics  
Licencia: GPL-3.0  

### Technický prehľad

YOLOv8 je moderná konvolučná neurónová sieť (CNN) určená na detekciu objektov v reálnom čase. Patrí do rodiny YOLO (You Only Look Once) a je optimalizovaná na:

- Vysokú rýchlosť detekcie  
- Dobré pomer presnosť/výkon  
- Jednoduché nasadenie  
- Podporu GPU  

Typické použitia:
- Detekcia ŠPZ  
- Sledovanie objektov  
- Priemyselná inšpekcia  
- Monitorovanie a bezpečnostné systémy  

### Architektonické vlastnosti

- Detekcia bez kotviacich boxov (anchor-free)  
- Extrakcia viacerých mierok (multi-scale)  
- Implementácia v PyTorch  
- Podpora exportu (ONNX, TensorRT, atď.)  

### Licenčné úvahy (kritické)

YOLOv8 je vydané pod **GPL-3.0**.

Dôsledky:

- Ak sa integruje do distribuovaného produktu, celý projekt musí byť v súlade s GPL  
- Povinnosť sprístupniť zdrojový kód pri distribúcii  
- Nevhodné pre uzavreté komerčné softvéry  

### Dlhodobá stabilita

- Aktívne udržiavané  
- Technicky pripravené na budúcnosť  
- Legálne obmedzené pre komerčné projekty  

### Odporúčanie

Použiť iba pri:
- Interných nástrojoch  
- Výskumných prostrediach  
- Open-source projektoch  

Nevhodné pri:
- Komerčných distribuovaných softvéroch  
- Proprietárnych produktoch

---

## RT-DETR (Hugging Face, Garon16)

Model: https://huggingface.co/Garon16/rtdetr_r50vd_russia_plate_detector  
Licencia: Apache-2.0  

### Technický prehľad

RT-DETR (Real-Time Detection Transformer) je transformer-based model na detekciu objektov odvodený od architektúry DETR.

Kľúčové vlastnosti:

- Transformer encoder-decoder štruktúra  
- Žiadne anchor boxy  
- End-to-end detekcia  
- Moderná architektúra v súlade s aktuálnymi trendmi  

Pretrained model príklad:
- Detekcia ruských ŠPZ  

### Licencia

Apache-2.0:

- Povolené komerčné použitie  
- Povolené úpravy  
- Bez copyleft požiadaviek  
- Bezpečné pre proprietárne produkty  

### Architektonické silné stránky

- Transformerová architektúra (SOTA trend)  
- Silná komunita a ekosystém  
- Jednoduchá integrácia cez Transformers API  
- Dlhodobá životaschopnosť  

### Riziká

- Konkrétny pretrained model môže prestať byť aktualizovaný  
- Základná architektúra a ekosystém sú stabilné  

### Odporúčanie

Ideálne pre:
- Komerčné systémy  
- Dlhodobé projekty  
- Licenčne bezpečné nasadenie

---

# 2. Modely na detekciu tvárí

## MTCNN (Multi-Task Cascaded CNN)

Repository: https://github.com/ipazc/mtcnn  
Licencia: MIT  

### Technický prehľad

MTCNN je kaskádová CNN na detekciu tvárí:

- P-Net (proposal)  
- R-Net (refinement)  
- O-Net (output + landmarky)  

Funkcie:
- Detekcia tvárí  
- Extrakcia landmarkov  
- Ľahká inferencia  

### Silné stránky

- Permisívna licencia MIT  
- Jednoduchá integrácia  
- Dobré pre kontrolované prostredia  

### Slabé stránky

- Staršia architektúra  
- Nie SOTA podľa moderných štandardov  
- Pomalšie ako optimalizované DNN detektory  

### Odporúčanie

Použiť keď:
- Potrebné landmarky tváre  
- Stredná presnosť postačuje  
- Jednoduchosť nad SOTA presnosťou

---

## OpenCV DNN Face Detector (Caffe / ResNet SSD)

Licencia: BSD  

### Technický prehľad

Predtrénovaný DNN detektor tvárí používa:

- ResNet SSD  
- Caffe model  
- Integrácia cez OpenCV DNN modul  

Vlastnosti:

- Nie je potrebný ťažký ML framework  
- CPU-friendly  
- Produkčne stabilný  

### Silné stránky

- Extrémne stabilné  
- Minimálne závislosti  
- Priemyselná spoľahlivosť  
- Permisívna BSD licencia  

### Slabé stránky

- Menej flexibilné pre vlastný tréning  
- Nie SOTA presnosť  

### Odporúčanie

Najlepšie pre:
- Dlhodobé enterprise systémy  
- Prostredia s minimálnymi závislosťami  
- Embedded nasadenia

---

## Haar Cascade Classifier

Licencia: BSD  

### Technický prehľad

Klasický Viola–Jones detektor používa:

- Haar-like features  
- Integral images  
- Boosted cascades  

Vlastnosti:

- Veľmi ľahký  
- CPU-only  
- Extrémne rýchly  

### Limitácie

- Slabý výkon v zložitých podmienkach  
- Citlivý na svetlo a uhly  
- Zastaralý pre moderné aplikácie  

### Odporúčanie

Použiť iba keď:
- Hardvér je extrémne obmedzený  
- Nízke požiadavky na presnosť  
- Legacy kompatibilita potrebná

---

# 3. OCR

## EasyOCR

Repository: https://github.com/JaidedAI/EasyOCR  
Licencia: Apache-2.0  

### Technický prehľad

EasyOCR poskytuje:

- Multijazykové OCR  
- Deep learning-based rozpoznávanie  
- PyTorch backend  
- Jednoduché API  

Podporované:
- ŠPZ  
- Dokumenty  
- Text v scéne  

### Silné stránky

- Apache-2.0 licencia  
- Jednoduchá integrácia  
- Aktívna komunita  
- Komerčne použiteľné  

### Slabé stránky

- Nie najrýchlejší pri veľkom objeme  
- Limitovaná jemná kontrola modelu  

### Odporúčanie

Ideálne pre:
- Malé a stredné OCR projekty  
- Rozpoznávanie ŠPZ  
- Rýchle prototypovanie

---

# 4. Core Frameworky

## TensorFlow

Web: https://www.tensorflow.org  
Licencia: Apache-2.0  

- Zrelé riešenie  
- Produkčne pripravené  
- Dlhodobo udržiavané  
- Enterprise-friendly

---

## PyTorch

Web: https://pytorch.org  
Licencia: BSD  

- Výskumno-priateľské  
- Flexibilné  
- Široko používané  
- Bezpečné pre komerčné použitie

---

## Transformers (Hugging Face)

Dokumentácia: https://huggingface.co/docs/transformers  
Licencia: Apache-2.0  

Poskytuje:
- AutoModelForObjectDetection  
- AutoImageProcessor  
- Jednotné API pre načítanie modelov  

Silná komunita a dlhodobá stabilita.

---

# 5. Podporné knižnice

## NumPy

Web: https://numpy.org  
Licencia: BSD  

- Kľúčová knižnica pre numerické výpočty  
- Fundamentálna závislosť  
- Extrémne stabilná

---

## scikit-learn

Web: https://scikit-learn.org  
Licencia: BSD  

- Vyhodnocovacie metriky  
- Precision / Recall  
- Nástroje pre validáciu modelov

---

# Finálne architektonické hodnotenie

## Najbezpečnejší long-term komerčný stack

Detekcia:
RT-DETR (Apache-2.0)  

Framework:
PyTorch (BSD)  

Predspracovanie:
OpenCV (BSD)  

OCR:
EasyOCR (Apache-2.0)  

Vyhodnotenie:
NumPy + scikit-learn (BSD)

---

## Zhrnutie rizík

Nízkorizikové:
- PyTorch  
- OpenCV  
- NumPy  
- scikit-learn  
- TensorFlow  
- EasyOCR  
- RT-DETR (Apache)

Stredné riziko:
- MTCNN (starnúca architektúra)

Vysoké právne riziko:
- YOLOv8 (GPL-3.0)

---

# Záverečné odporúčanie

Pri budovaní komerčného, dlhodobo podporovaného systému:

Odporúčaný stack:  
**RT-DETR + PyTorch + OpenCV + EasyOCR**

Pri prioritizácii maximálnej stability a minimálneho licenčného rizika:  
**OpenCV DNN + EasyOCR**

Pre interné R&D použitie:  
YOLOv8 je akceptovateľné.

---

Pripravené s ohľadom na architektúru, licenciu a dlhodobú udržateľnosť.
