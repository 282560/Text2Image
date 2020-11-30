# Text2Image - Python

Repozytorium zawierające *projekt bazowy*, czyli ten człon pracy magisterskiej, który odpowiada za trening i testy oraz generowanie modeli.

## Przed przystąpieniem do treningu/testu

Należy wpierw zamaskować wszystkie inne procesory graficzne (_GPU_), jeżeli maszyna posiada więcej niż jeden. Robi się to po to, by inni użytkownicy również mogli z nich korzystać. Służy do tego komenda:
```
set CUDA_VISIBLE_DEVICES={numer karty}
```

## Instrukcja

Pomoc wyświetla się z katalogu głównego projektu (tam gdzie znajduje się plik `runtime.py`). Wówczas wystarczy wpisać:
```
python runtime.py -h
```
lub
```
python runtime.py --help
```
Ale pomoc jest również opisana poniżej.
Aby poprawnie skorzystać ze skryptu należy wprowadzić komendę:
```
python runtime.py {operation} {dataset} {learning_rate} {batch_size} {epochs}
```
np.
```
python runtime.py train flowers 0.0002 100 1000
```
aby uruchomić _trening_ na na zbiorze _kwiatów_, ze stałą uczenia równą _0.0002_, rozmiarem paczki _100_ oraz na czas _1 000_ epok.
Tak więc:

* `operation` – rodzaj operacji, według której dane będą przetwarzane (możliwe opcje to `train` lub `test`).
* `dataset` – nazwa zbioru danych (zdefiniowane jest 5 nazw: `flowers`, `birds`, `three_flowers`, `three_birds` oraz `three_fruits`).
* `learning_rate` – wartość stałej uczenia (wartość rekomendowana przez autorów sieci to `0.0002`).
* `batch_size` – rozmiar paczki danych jakimi będą pobierane próbki.
* `epochs` – ilość epok.

*Uwaga! Wszystkie wyżej wymienione parametry są parametrami obowiązkowymi.*

## Puste pliki

W niektórych katalogach znajdują się puste pliki. Służą one jedynie zachowaniu struktury plików w repozytorium _GitHub_.

## Informacje

Autor: Cezary Pietruszyński

Promotor: dr Marek Grochowski