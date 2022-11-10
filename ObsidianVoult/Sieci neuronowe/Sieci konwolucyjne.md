#Konwolucyjne
Polegają na analizowaniu obrazu poprzez przesuwanie okna (detektora) po obrazie i wyszukiwaniu konkretnych cech.
![[Pasted image 20221103162546.png]]

Rozmiary filtrów dla sieci konwolucyjnych: **3x3**, **5x5**, 7x7, 9x9, 11x11.
![[Pasted image 20221103162705.png|400]]

```ad-note
title: Filtr
Wielkość filtru (detektora) jest jednym z hiperparametrów sieci konwolucyjnej

```

![[1_ulfFYH5HbWpLTIfuebj5mQ.gif|300]]

Sieci konwolucyjne uczy się tak samo jak sieci fit forward czyli z wykorzystaniem graientu. Parametrami uczenia sieci są wagi i biasy. Oprócz tego sieci konwolucyjne posiadają zestaw wielu hiperparametrów
**Hiperparametry:**
- wielkość okna (filtra) (kernel)
- krok (stride)
- liczba filtrów
- ramka (pading)
- współczynik uczenia
- sposób inicjalizacji wag

W przypadku sieci konowlucyjnych w celu nauczenia sieci potrzebne jest znacznie mniej wag niż w przypadku seci płaskich, gęsto połączonych ponieważ wagi dla jednego kanału są współdzielone i są związane z wielkością filtra.

```ad-info
Hiperparametry, w odróżnieniu od parametrów nie są ustalane w procesie uczenia sieci (stąd ich inna nazwa). Dobierane są na podstawie eksperymentów i doświadczenia
```


### Pooling (subsampling)
Zmniejszanie wymiarów map cech w kolejnych warstwach siecie poprzez wybranie jednej wartości występującej w oknie.
**Max pooling** - wybranie wartości maksymalnej z okna
**Average pooling** - wybranie wartości średniej z okna

![[Pasted image 20221110153358.png]]

### Dropout
Wyłączenie części neuronów podczas uczenia.

### Różnice w propagacji wstecznej
W sieciach konwolucyjnych, jeżeli rozrysować je w taki sposób jak przedstawia się sieci zwykłe, można zauważyć że neurony wyjściowe nie są w pełni połączone z neuronami wejściowymi.
![[Pasted image 20221110155119.png|600]]
Zmiana wag odbywa się w sposób inny niż w sieciach MLP
$$z^{l+!}_{x,y} = w^{l+1} * \sigma(z^l_{x,y}) + b^{l+1}_{x,y} = \sum_a \sum_b w^{l+1}_{a,b}\sigma(z^l_{x-a, y-b}) + b^{l+1}_{x,y}$$
Reguła łańcuchowa:
$$\frac{\partial C}{\partial z^l_{x,y}} = \sum_{x'} \sum_{y'}\frac{\partial C}{\partial z^{l+1}_{x',y'}}\frac{\partial z^{1+l}_{x',y'}}{\partial z^l_{x,y}} = \sum_{x'} \sum_{y'} \delta^{l+1}_{x',y'} \frac{\partial (\sum_a \sum_b w^{l+1}_{a,b}\sigma(z^l_{x-a, y-b}) + b^{l+1}_{x,y})}{\partial z^l_{x',y'}}$$

**Algorytm**
- Zasotsuj wzorzec wejsciowy do warstwy wejściowej
- Dla każdej warstwy $l = 1, 2, ...$ oblicz $z$ oraz $a$
- Oblicz błąd w warstwie wyjściowej
- Rzutuj błąd w wstecz dla warstw $l$
- przyrost wagi $\Delta w$ jest proporcjonalny do składowej granidntu


