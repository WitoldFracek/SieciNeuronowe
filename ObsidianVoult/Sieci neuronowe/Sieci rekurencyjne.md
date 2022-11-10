Sprawdzają się w przypadku gdy wartości które te sieci mają za zadanie przewidzieć, są zależne od wcześniej występujących wartości.

```ad-question
title: Dlaczego potrzebujemy sieci rekurencyjnych
Sieci fit forward chociaż osiągają dobre wyniki dla klasyfikacji nie sprawdzają się dla danych będących ze sobą w relacji na przestrzeni czasu (serie czasowe). Stąd właśnie powstała potrzeba na sieci rekurencyjne

```

Sieci rekurencyjne mają podobną budowę co sieci fit forward ale posiadają cykliczne połączenia pomiędzy neuronami.
![[Pasted image 20221110162240.png]]
![[Pasted image 20221110162302.png]]
Połaczenia cykliczne mogą występować również pomiędzy neuronami z różnych warstw.
Połączenia neuronów samych do sebie nazywa się sprzężeniem.

Wyliczanie aktywacji:
$$h^{(t)} = \sigma(W^{hx}x^{(t)} + W^{hh}h^{(t-1)} + b_h)$$
$$\hat{y}^{(t)} = softmax(W^{yh}h^{(t)} + b_y)$$
Żeby ułatwić obliczanie sieci zamiast połączeń cyklicznych stosuje się nowy model przyjmujący jeden zestaw cech więcej. Czylu dla kroku zerowego na wejśiu sieć dostaje jedynie dane $x_0$ a w kroku pierwszym dostaje $x_1$ oraz $h_0$. Postać taką nazywa się postacią rozwiniętą sieci.

![[Pasted image 20221110163346.png]]

Przy obliczaniu aktywacji najpierw idzie się do przodu w czasie a później dopiero do kolejnej warstwy. Dopiero po obliczeniu ostaniej warstwy w ostaniej chwili czasu można obliczyć błąd a następnie rzutować gradienty w koerunku przeciwnym do aktywacji.

### Podsumowanie
- Pozwalają elstycznie zaprojektować architektórę
- Klasyczne RNN sąproste, ale nieefektywne
- Często pojawią się eksplozja gradnietu albo znikajacy gradient
- Ekspodujący gradient jest łatwiejszy do rozwiązania  (obcięcie wag)
- Zanikający gradient rozwiązywany jest przez nowe archtektury: LSTM lub GRU.
![[Pasted image 20221110165005.png]]

**LSTM** - Long Short Term Memory
Sigma jest bramką i jest zazwyczaj powiązana z funkcją aktywacji. Bramki mają zapobiegaćzanikowi gradientu.
Górna czarna linia ozacza czas przetwarzania.
Bramki:
- forget gate - bierze $h_{t-1}$ i produkuje wetor wartości z zakresu (0, 1), który mówi jak bardzo stan będzie zmieniony. 
	$f_1 = \sigma(W_f [h_{t-1, x_t}] + b_f)$
- write - decyduje która wielkość będzie uaktualniana. Jest implementowana jako funkjca sigmoidalna.
	$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ 
	$\hat{C}_t = tanh(W_C [h_{t-1}, x_t] + b_C$
- Output gate - obliczanie wyjścia z komórki
	$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
	$h_t = o_t * tanh(C_t)$

**GRU** - Gated Recurrent Unit
Zamiast 3 bramek jak w LSTM wykorzystuje 2 bramki (update i reset). Nie ma wyjściowej brami, nie ma drugiej nieliniowości. Nie korzysta z pamięci $C$, redukuje się do stanu ukrytego GRU
$z_t = \sigma(W_z [h_{t-1}, x_t])$
$r_t = \sigma(W_r [h_{t-1}, x_t])$
$\tilde{h}_t = tanh(W [r_t * h_{t-1}, x_t])$
$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$

### Możliwe zastosowania
- Generowanie tekstu
- Translacja maszynowa (ang -> pl)
- Transkrypcja mowy na tekst
- Przetwarzanie rysunku
- Generowanie podpisów do obrazków

Love alters not with his brief hours and weeks