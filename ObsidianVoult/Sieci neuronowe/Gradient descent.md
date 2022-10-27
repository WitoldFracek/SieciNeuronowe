Minimalizuje funckje kosztu $J(\theta)$ parametryzowaną przez parametr modelu $\theta \in R^d$ . Należy do określania parametrów w procesie ich iteracyjnego uaktualniania w kierunku przeciwnym do gradientu $\nabla_{\theta}J(\theta)$ . Współczynnik $\eta$ określa rozmiar kroku używany w celu osiągnięcia minimum błędu.

![[Pasted image 20221027160756.png]]

**Aktualizacja:**
$$\theta := \theta - \eta \nabla_{\theta}J(\theta)$$
```ad-info
title: Ważne
Metoda gradientowa pozwala na znalezienie optimum lokalnego co nie znaczy że osiągnięto optimum globalne. ALgorytm może "utknąć" w znalezionym optimum lokalnym
```


![[Pasted image 20221027160939.png]]

