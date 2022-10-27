27.10.2022

#### Sieci neuronowe - dobre praktyki
#### Kodowanie wartości wyliczeniowych (enumeracji)
Na kodowanie wykrozytujemy tyle neuronów ile jes wartości wyliczeniowych. Na przykład dla miesięcy weźmiemy 12 neuronów na wejściu jako osobnych cech (uzupełniane na przykład 0, 1).

#### Zmienne o małym zróżnicowaniu
W przypadku wartości które niewiele różnią się od siebie lepiej jest na przykład podawać przyrosty wartości (na przykład wartość dolara zamiast podawać 4.43 a następnie 4.44 to podajemy przyrost czyli 0.01).

#### Ocena statystyk zbioru
Dobrze jest sobie zwizualizować zbiór danych wejściowych żeby mieć pewność że cechy są dobrze dobrane.
![[Pasted image 20221027153528.png |]]

#### Trenowanie i walidacja
![[Pasted image 20221027153628.png]]

#### Regularyzacja
Do funkcji kary dodaje się kolejny człon jako kara.
**Regularyzacja L1**
$$C = C_0 + \frac{\lambda}{p}\sum_w{|w|}$$
$$w(t + 1) = w(t) - \eta\frac{\partial C_0}{\partial w} - \frac{\eta \lambda}{p}sign(w)$$

**Regularyzacja L2**
$$C = C_0 + \frac{\lambda}{p}\sum_w{w^2}$$
$$w(t+1) = w(t) - \eta \frac{\partial C_0}{\partial w} - \frac{\eta \lambda}{p}w$$

```ad-warning
title: BIas
Bias zostaje bez regularyzacji

```


#### Dropout
- Zapobiega wspólnej adapracji neuronów (detektorów cech w warstwie ukrytej).
- Możemy patrzeć na taką sieć jak na zestaw sieci (zespół klasyfikaotrów).
- Szczególnei istotny dla głębokich modeli ze względu na większy problem z przetwarzaniem danych.

#### Powiększanie zbioru
- Rotacje
- Rozmazanie
- Kontrast
- Skalowanie
- Rozjaśnienia
- Transformacja projekcji

#### Wczesne zatrzymywanie uczenia
Należy cały czas sprawdzać jak obeny stan sieci sprawdza się na zbiorze walidacyjnym i w momencie kiedy sieć zaczyna sobie gorzej radzić oznacza to że się przeucza. Wtedy jeśli widzi się że błąd zaczyna znacząco rosnąć to należy przerwać uczenie sieci.
![[Pasted image 20221027154720.png|500]]

#### Sposoby oceny jakości sieci
[[Krzywa AUC-ROC]]
[[Macierz pomyłek]]

#### Optymalizacja gradientowa
[[Gradient descent]]

#### Ustalanie współczynnkia uczenia
**Stały współczynnik uczenia**
- Rzadko stosowany
- Najmniej efektywny
- Na podstawie badań empirycznych można przyjąć że:
$$\mu \le min(\frac{1}{n_{in}})$$
gdzie $n_{in}$ oznacza wejście k-tego neuronu w warstwie.

**Adaptacyjny współćzynnik uczenia**
W tej metodzie na podstawie porónywania sumarczynego błędu $\epsilon$ w i-tej iteracji z jego poprzednią wartością, określa się strategię zmian współczynnika uczenia.
W celu przyspieszenia uczenia współczynnik jest zwiększany, sprawdzając czy błąd nie zaczyna rosnąć. Dopuszcza się przy tym nienaczny wzrost błędu.
**Oznaczenia:**
$\mu_i$ - wartość współczynnika w i-tej iteracji
$\epsilon_i$ - wartość błędu.
$\rho_d$ - współczynnik zmniejszania współćzynnika uczenia
$\rho_i$ - współczynnik zwiększania współćzynnika uczenia

Jeśli $\epsilon_i \gt k_w\epsilon_{i-1}$ to $\mu_{i} = \rho_d \mu_{i-1}$
Jeśli $\epsilon_i \le k_w\epsilon_{i-1}$ to $\mu_{i} = \rho_i \mu_{i-1}$

#### Momentum
Używając momentum dodajemy dobierzącego wekotra wag część $\gamma$ zmian wag kroku poprzedniego.
**Momentum klasyczne** 
$$v_t = \gamma v_{t - 1} + \eta \nabla_\theta J(\theta)$$
$$\theta := \theta - v_t$$
$\gamma$ ma zazwyczaj wartości między 0.8 a 0.9

**Momentum Nestora**
$$v_t = \gamma v_{t - 1} + \eta \nabla_{\theta}J(\theta - \gamma v_{t-1})$$
$$\theta := \theta - v_t$$

![[Pasted image 20221027162811.png]]

```ad-warning
title: Ważne
Jeżeli używamy współćzynnika momentum powinniśmy zmniejszyć współćzynnik uczenia
```

#### Optymaliztor Adagrad
**Oryginalna aktualizcja parametrów:**
$$\theta_t \leftarrow \theta_{t - 1} - \eta \nabla C(\theta_{t - 1})$$
**Teraz każda waga jest rozważana oddzielnie:**
$$w_{t + 1} = w_t - \eta_w g_t $$
$$g_t = \frac{\partial C(\theta_t)}{\partial w}$$
$$\eta_w = \frac{\eta}{\sqrt{\sum_{i=0}^{t}(g^i)^2}}$$

```ad-warning
title: Ważne
Wartości ze zwględu na sumę w mianowniku mogą stać się bardzo małe. Jeżeli suma gradnientów będzie bardzo duża współczynnik uczenia będzie bardzo mały

```
**Korzyści:**
- Eliminuje potrzebę ręcznego ustawiania współczynnika uczenia. Trzeba ustalić jedynie jego początkową wartość.

#### Optymalizator Adadelta
Pozbywa się problemu optymalizatora Adagrad poprzez określenie dopuszczalnego minimalnego okna dla sumy gradientów przez co nie bedzie ona zmniejszać za bardzo współczynnika uczenia.
Zamiast przetrzymywania przeszłych gradientów suma gradientów jest rekursywnie  definiowana jako zmniejszająca się średnia wszystkich przeszłych gradientów (decaying average of all past squared gradients).

$$E[g^2]_t = \gamma E[g^2]_{t - 1} + (1 - \gamma)g_t^2$$

W czystym [[Gradient descent|SGD]] mamy:
$$\theta := \theta - \eta \nabla_{\theta}J(\theta)$$
W Adagrad mamy:
$$\Delta \theta_t = - \frac{\eta}{\sqrt{G_t + \epsilon}} \circledcirc g_t$$
Zastępujemy $G_t$ przez $E[g^2]_t$ :
$$\Delta \theta = - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}$$
Root mean squared (RMS):
$$\Delta \theta = -\frac{\eta}{RMS[g]_t}g_t$$
### Inicjalizacja wag
**Randomizacja w zakresie** $[-{a}/{\sqrt{n_{in}}}, {a}/{\sqrt{n_{in}}}]$ 
a jest tak dobrane , żeby wariancja wag odpowiadała punktowi maksymalnej krzywizny funkcji aktywacji (dla standardowej sigmoidy 2.38)

**Randomizacja wag w zakresie** $[-2/{\sqrt{n_{in}}}, 2/{\sqrt{n_{in}}}]$
$n_{in}$ - liczba wejść do danego neuronu

**Randomizacja wag w zakresie** $[-\sqrt[n_{in}]{N_h}, \sqrt[n_{in}]{N_h}]$
$N_h$ - liczba neuronów w warstwie ukrytej

**Randomizacja w warstwie wejściowej**
Losowo w zakkresie $[-0.5, 0.5]$