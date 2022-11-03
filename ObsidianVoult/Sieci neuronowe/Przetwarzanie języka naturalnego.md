### Word 2 vec - słowa jako wektory
**Kiedyś:**
Zapisywanie słów jako pozycja w słowniku. Długośćwekora to długość słownika i jedynka na miejscu w którym wystąpiło słowo.
**Nazwa reprezentacji:** worek słów (bag of words)
**Problemy:**
- klątwa wymiarowości - duże wektory dla dużych zbiorów słów.
- Utrata wymiarowości - małe powiązanie słów między sobą

**Obecnie:**
Opracowanie metody reprezentacji słów jako wektorów o mniejszej niż wcześniej wymiarowości ([[Embedding|embeding]] - reprezentacja wektorowa)

#### Reprezentacje:
**Reprezentacja CBOW** - continous bag of words
Słowa są reprezentowane jako wektory słownikowe (jedynka na tym miejscu na któym występuje słowo w słowniku). Na podstawie sąsiedztwa słowa staramy się przewidzieć jakie słowo ma wystąpić.
Składa się z jednej ukrytej **warstwy liniowej**.

**Reprezentacja Skip-Gram**
Wyrzuca się jedno słowo i patrzy które słowa mogą być w *otaczającym sąsiedztwie*.
Odwrotne przenaczenie niż CBOW. Ma za zadanie na podstawie słowa przedstawić listę słów znajdujących się w sąsiedztwie.
Składa się z jednej ukrytej **warstwy liniowej**.

![[Pasted image 20221103154650.png]]

Ta pojedyncza **warstwa liniowa** staje sie [[embeding|embedingiem]] dla danego słowa. Jest mniejwywmiarowym wektorem niż wektor na wejściu i kładają się z liczb rzeczywistych i stanowią pewną reprezentację ukrytą. Słowa o podobnym znaczeniu powinny mieć podobną reprezentację ukrytą.

![[Pasted image 20221103154130.png|500]]

