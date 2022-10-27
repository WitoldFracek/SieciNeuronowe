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
