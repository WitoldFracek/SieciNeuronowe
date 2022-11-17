# AlexNET 2012
Wytrenowana na zestwie ILSVRC  (1000 classes) przez tydzień.
Miała 5 warstw konwolucyjnych i 3 warstwy w pełni połączone:
- Warstwy 1, 2, 5 miały max-pooling
- Dropout na dwóch ostatnich w pełni połączonych sieciach
- Warstwy 1 i 2 przeprowadzały normalizację
- Kolejne warstwy tworzyły hierarchię cech (najpeirw plamy i bloby, potem złożone kształty, na końcu przewidywane klasy (sematnic attributes))

### Warstwa 1: Gabor and color blobs
![[Pasted image 20221117152800.png]]

### Reperezentacja rozpoznawania chech na zdjęciach
![[Pasted image 20221117153037.png]]

# VGG 2014
### Visual Geometry Group
Miała około 160M parametrów i 16 warstw. W odróżnieniu od sieci AlexNET używała filtrów o mniejszych maskach (zamiast 11x11 i 5x5 w późniejszych warstwach, używała filtów 3x3)

# GoogleNet
Dużo bardziej złożóna struktura. Składała się z modułów.
![[Pasted image 20221117153638.png]]

<mark class="hltr-blue">Konwolucje</mark>  <mark class="hltr-red">Pooling</mark>  <mark class="hltr-yellow">Softmax</mark>  <mark class="hltr-green">Inne</mark>
![[Pasted image 20221117153557.png]]

#### Konkatenacja warstw sieci
Żeby połączyć warstwy o różńych rozmiarach tensorów stosuje się ramki uzupełnijące brakujące rozmiary tam gdzie jest to potrzebne.
![[Pasted image 20221117154053.png|400]]

# ResNet
Posiadała połączenia resydualne które pozwalają na szybsze uczenie sieci. Dzięi temu można w krószym czasie wytrenować sieci o znacznie większej liczbie warstw. Połącznie resyduwalne może przeskakiwać kilka warstw.
![[Pasted image 20221117154152.png|400]]





