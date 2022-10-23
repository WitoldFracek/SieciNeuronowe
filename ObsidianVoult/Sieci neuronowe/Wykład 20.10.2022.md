#Architektura #DaneWejsciowe #Przeuczanie
### Projektowanie architektury sieci
**architektura** - sposób połączneia neuronów, liczność neuronów
**Warstwy:**
	- warstwa wejściowa:
		- wielkość równa licznbie cech badanych
	- wastwa wjściowa:
		- wielkść równa liczbie wzorców do klasyfikacji (dla *n* klas mamy *n* neuronów na wyjściu)

##### Dane
Kiedy mamy mało zdjęć do warstwy wejściowej należy rozpatrzeć dane pod kontem konkretnyc hcech. Na przykad gdy chcemy rozpoznawać emocje na twarzy a mamy mało zdjęć twarzy możemy uwzględnić cechy takie jak: rozwarcie ust, kąt rozwarcia, wysokość brwi, czy brwi są na jednej lini, szerokość źrenic, itp. W ten sposób ograniczamy elementy związane z uczeniem do mniejszej liczby cech przez co łatwiej taki model nauczyć w przypadku ograniczonego zbioru danych wejściowych.

#### Skład danych
- Zbiór treningowy - zbiór na któym uczy się model
- Zbiór walidacyjny (dev - development)
- Zbiór testowy - do oceny końcowej jakości modelu

##### Dlaczego powinniśmy skalowaćwejście
Przykład: udzielanie pożyczki w banku.
Analizujemy:
	- Zarobek (wartości: 2 500 - 50 000)
	- Liczba osób na utrzymaniu (wartości: 1 - 8)
	- Wiek osoby biorącej kredyt (wartości 18 - 70)

Opis klienta: \[3 000, 2, 25]
Największy wpływ na aktywację będzie miała wartość największa czyli w tym wypadku zarobek.
```python
x = np.array([3_000, 2, 25])
activation = w.dot(x) #  wartości aktywacji duże dla dużych danych wejściowych
```

Żeby uniknąć takich błędów należy przeskalować dane.
Sposoby skalowania:
	Najprościej:
		$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$
	Ze średnią:
		$$x' = \frac{x - x_{mean}}{x_{max} - x_{min}}$$
	Z odchyleniem standardowym:
		$$x' = \frac{x - x_mean}{\sigma}$$
	Normowanie do jedynki: (dane jako wektor)
		$$x' = \frac{x}{||x||}$$
#### Wielkość warstw ukrytych
Są różne sposoby ustalenia liczby neuronów w warstwach ukrytych
**Średnia arytmetyczna**:
$$n_n = \frac{1}{2} (n_{in} + n_{out})$$
**Średnia geometryczna**
$$n_n = \sqrt{n_{in} * n_{out}} $$
#### Dobór współczynnika uczenia
**Zbyt <mark class="hltr-cyan">mały</mark> współczynnik uczenia** - długie oczekiwanie na jakikolwiek efekt.
**Zbyt <mark class="hltr-red">duży</mark> współćzynnik uczenia** - brak zbierzności uczenia.

```ad-note
Metody gradientowe powodują znalezienie minimum lokalnego co nie zawsze oznacza że uda się znaleźć minimum globalne. Jeżeli sieć nie jest w stanie osiągnąć zadanego minimum to losuje się wagi od początku z nadzieją na trafienie na lepszy start.
```


#### Trenownaie i walidacja modelu
Trenujemy sieć dobierając wartości hiperparametrów.
**Hiperparametry**:
- liczba neuronów w warstwach
- Współczynnik uczenia
- Zakres inicjalizacji wag początkowych

#### Sposoby unikania przeuczenia
Dla mniejszego modelu trzeba znaleźć mniej parametrów zatem dostajemy mniejszy koszt obliczeniowy. Istnieje większa szansa na zdolność do uogulniania.
Pruning sieci - usuwanie nieistotnych połączeń. Można wyeliminiować połączenia dla których wrażliwość na zmiany danych jest mała.
![[Overfitting.png|200]]
**Regularyzacja** - wprowadzanie kary do funkcji kosztu:
- <mark class="hltr-cyan">regularyzacja L1</mark> - suma wartości bezwzględnych wszystkich wag
	$$c = c_0 + \frac{\lambda}{p}\sum{|w|}$$
	Gdzie:
	$c_0$ - wylicozy wcześniej koszt
	$\lambda$ - współczynnik regularyzacji
	$p$ - liczba wszystkich wzorców

	Wtedy zmianę wag można predstawić jako:
	$$w(t+1) = w(t) - \eta \frac{\partial c_0}{\partial w} - \frac{\eta \lambda}{p} sign(w)$$
	Gdzie:
	$\eta$ - 
	$w$ - wagi warstwy

- <mark class="hltr-cyan">regularyzajca L2</mark> - suma kwadratów wartości wszystkich wag
	$$c = c_0 + \frac{\lambda}{2p}\sum{w^2}$$
	Gdzie:
	$c_0$ - wylicozy wcześniej koszt
	$\lambda$ - współczynnik regularyzacji
	$p$ - liczba wszystkich wzorców

	Wtedy zmianę wag można predstawić jako:
	
$$w(t+1) = w(t) - \eta \frac{\partial c_0}{\partial w} - \frac{\eta \lambda}{p} w$$

**Dropout** - Wycina się pewne neurony (ustawia się wagi na 0). Jest to czasowe wyłączenie neuronów. Wyłącza się je na przykład dla jednej paczki danych. Powoduje to że przy trenowaniu każda próbka działa na innego rodzaju sieci a ze względu na brak jakiegoś neuronu sieć jest mniejsza więc osiąga większą wrażliwość ze względu na mniejszą pojemność.

**Sztuczne powiększanie zbioru uczącego**:
- Dodanie szumu do danych (proste we wzorcach obrazowych i dźwiękowych)
- Dla obrazów możemy użyć transformacji (translacja, obrót, odbicie, rozmycie, zmiana skali)

**Wczesne zatrzymywanie uczenia (early stopping):**
Co iterację waliduje się obecne nauczenie zbiorem walidacyjnym. W momencie gdy wartość błędu zaczyna się zaikszać znaczy że model się przeucza i należy przerwać proces.

#### Sprawdzanie zdolności do uogólniania
*Można to zrobić dla odpowiednich warunków*
**Walidacja krzyżowa** - dla małych zbiorów (k-cross validation). Polega na podzieleniu zbioru danych na *k* podzbiorów. Sieć uczy się na *k-1* podzbiorach a na *k-tym* podzbiorze się testuje.
**Hold one out** - stosowana gdy mamy bardzo mało wzorców do uczenia. Podobnie jak w przypadku walidacji krzyżowej ale ze zbioru wyłączamy tylko jedną próbkę a nie pewien podzbiór.
**Bootstraping** - towrzymy podzbiory zbioru danych przez losowanie ze zwracaniem (dotyczy zarówno zbioru testowego i uczącego).

#### Maczierze pomyłek
|    |positive|negative|
| ----------|--------|--------|
| positive | True Positive | False Negative |
| negative | False Positive  | True Negative |

**Miara oceny jakości** - mówi o tym jak dobrze model unika złej klasyfikacji.
$$recall = \frac{TP}{TP + FN}$$
**Specyficzność** - mówi o tym jak dobrze model unika False Positives.
$$specificity = \frac{TN}{TN + FP}$$
**Dokładność**
$$accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
**Prezycja** - mówi o stopniu powtarzalności wyniku.
$$precission = \frac{TP}{TP + FP}$$




