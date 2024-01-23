# Kod

config:
  - config.yaml - parametry całego projektu tzn. modelu, uczenia, danych itp.

data:
  - MFCC - folder zawierający wyekstrahowane cechy MFCC zarówno ze zbioru treningowego, jak i testowego
  - TEST - oryginalne dane testowe TIMIT
  - TRAIN - oryginalne dane treningowe TIMIT
  - test_data.csv - oryginalny spis danych testowych TIMIT
  - train_data.csv - oryginalny spis danych treningowych TIMIT

files:
  - core_test_set.csv - spis wybranych danych testowych
  - core_train_set.csv - spis wybranych danych treningowych
  - core_train_subset.csv - spis podzbioru danych treningowych użyty do kontrolowania procesu uczenia
  - original_test_set.csv - spis całego zbiory testowego
  - original_train_set.csv - spis całego zbiory treningowego
  - tokenizer.json - plik przypisujący każdemu znakowi odmiennego indeksu

rnnt:
  - beam_search.py - dekodowanie przeszukiwaniem ścieżki
  - data.py - przetwarzanie nagrać na cechy MFCC, iterator danych
  - decoder.py - moduł dekodera architektury RNN-T
  - encoder.py - moduł enkodera architektury RNN-T
  - model.py - moduł warstwy łączącej, połączenie wszystkich modułów w finalną architekturę RNN-T, dekodowanie zachłanne
  - optim.py - dostępne algorytmy optymalizacji
  - recognizer.py - aplikacja okienkowa do wykrywania słów kluczowych
  - search.py - plik służący do naocznego testowania modelu, zapisuje efekt predykcji modelu do pliku wraz z prawidłową transkrypcją
  - test_system.py - testuje system wykrywania słów kluczowych, zwraca odpowiednie metryki oraz macierze pomyłek
  - tokenizer.py - zamienia znaki na liczby i na odwrót
  - train.py - trenuje model, testuje CER podczas trenowania, główna funkcja 'main()', która ładuje model i wywołuje wszystkie potrzebne funkcje
  - utils.py - funkcje pomocnicze np. inicjalizacja parametrów modelu, zapisywanie modelu itp.

timit/rnnt:
 - 2enc1dec_model.chkpt - wytrenowany model RNN-T

# Konfiguracja środowiska
Aby uruchomić kod należy zainstalować wiele zależności. Ponadto konieczne jest posiadanie systemu operacyjnego Ubuntu 18.04 lub 20.04. Najważniejsze z zależności/bibliotek to:
  - CUDA
  - Pytorch
  - warprnnt_pytorch (https://github.com/HawkAaron/warp-transducer) - należy usunąć folder warp_transducer, ściągnąć go i zainstalować dla swojej maszyny
  - torchaudio
  - PyQt5

Po zainstalowaniu powyższych należy doinstalować brakujące biblioteki sugerowane przez edytor.
