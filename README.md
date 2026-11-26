# DeepLearning2026 — Forecasting Vegetation Health from Satellite Time Series

Il progetto prevede l'evoluzione della vegetazione agricola a partire da tre osservazioni storiche Sentinel-2 *clear-sky*, meteo giornaliero storico e futuro, informazioni sulle nubi, calendario e classe climatica. Il meteo futuro è perturbato artificialmente per simulare l'incertezza delle previsioni e viene trattato come covariata esogena.

Il forecaster, ancora da implementare, dovrà stimare i quantili **0.1, 0.5 e 0.9** di NDVI, GNDVI ed EVI per le successive tre acquisizioni *clear-sky*, su un orizzonte medio indicativo di 14 giorni. L'output atteso ha forma `(batch, 3 step, 3 indici, 3 quantili)`, pari a **27 valori per finestra**.

La Bi-GRU e i loader sono disponibili. La Bi-GRU ricostruisce i tre indici storici e riceve **35 canali**, che diventano **43** dopo la concatenazione del climate embedding. Il checkpoint configurato è `history_imputer_weather_mask_best.pt`.

## Indici di vegetazione

| Indice | Formula | Informazione principale |
|--------|---------|------------------------|
| NDVI · Normalized Difference Vegetation Index | $\frac{NIR-Red}{NIR+Red}$ | Vigoria e copertura verde |
| GNDVI · Green Normalized Difference Vegetation Index | $\frac{NIR-Green}{NIR+Green}$ | Contenuto di clorofilla |
| EVI · Enhanced Vegetation Index | $2.5\,\frac{NIR-Red}{NIR+6Red-7.5Blue+1}$ | Monitoraggio della vegetazione densa con correzioni atmosferiche e del suolo |

## Dataset

Le serie derivano dai minicubi **GreenEarthNet** (Sentinel-2, copertura europea 2017–2022), integrati con osservazioni meteorologiche giornaliere **E-OBS**. I minicubi misurano 128 × 128 pixel con risoluzione di 20 m. Gli indici sono mediati sui pixel validi di classe *cropland*. Le finestre sono già segmentate, con tre acquisizioni storiche e tre future e stride pari a quattro. Il numero di giorni per sequenza varia con la disponibilità delle acquisizioni.

| Split | Directory sotto `DATA/` | Utilizzo |
|-------|------------------------|----------|
| Train | `train_multi_output_clear_sky/` | EDA, progettazione del preprocessing, fitting e validation interna |
| ID | `val_chopped_multi_output_clear_sky/` | Valutazione finale in-distribution |
| OOD-S | `ood-s_chopped_multi_output_clear_sky/` | Valutazione finale su aree non viste |
| OOD-T | `ood-t_chopped_multi_output_clear_sky/` | Valutazione finale su periodi non visti |

Lo split train locale contiene **84 aree e 104.699 finestre**, verificate sui file `.npz`. La traccia riporta 55.338 finestre per il train. Lo split di test nascosto è riservato ai docenti. [Documentazione del dataset](DATA/README.md) · [Traccia](DL_Project.pdf) · [Briefing del progetto](project_briefing.md)

Ogni area ha un file `<area>.npz` con le finestre nella scala originale e un file `<area>.json` con nomi delle feature e metadati. Le sequenze a lunghezza variabile sono archiviate in array NumPy di oggetti.

### Colonne di `history` e `future`

#### Variabili meteorologiche

| Indice | Feature |
|--------|---------|
| 0 | `wind` |
| 1 | `humidity` |
| 2 | `solar_irradiance` |
| 3 | `rainfall` |
| 4 | `pressure` |
| 5 | `avg_temperature` |
| 6 | `min_temperature` |
| 7 | `max_temperature` |

#### Informazioni sulla copertura nuvolosa

| Indice | Feature |
|--------|---------|
| 8 | `thick_clouds` |
| 9 | `thin_clouds` |
| 10 | `clouds_shadows` |

#### Informazioni temporali

| Indice | Feature |
|--------|---------|
| 11 | `day_of_year` |
| 12 | `day_of_year_fourier_sin_1` |
| 13 | `day_of_year_fourier_cos_1` |
| 14 | `day_of_year_fourier_sin_2` |
| 15 | `day_of_year_fourier_cos_2` |
| 16 | `day_of_year_fourier_sin_3` |
| 17 | `day_of_year_fourier_cos_3` |

#### Indici storici di vegetazione

| Indice | Feature |
|--------|---------|
| 18 | `target_history_avg_NDVI_clear_sky` |
| 19 | `target_history_avg_GNDVI_clear_sky` |
| 20 | `target_history_avg_EVI_clear_sky` |

Le colonne 18–20 di `future` sono intenzionalmente NaN e restano tali. I valori reali futuri sono in `target` e sono tutti disponibili nel train.

### Array per finestra

| Array | Descrizione |
|-------|-------------|
| `history` | Sequenza giornaliera delle osservazioni storiche |
| `history_mask` | Maschera dei valori storici mancanti |
| `history_timestamps` | Timestamp delle osservazioni storiche |
| `future` | Sequenza delle covariate future |
| `future_mask` | Maschera dei valori futuri mancanti |
| `future_noise` | Rumore associato alle covariate future |
| `future_timestamps` | Timestamp delle covariate future |
| `target` | Valori reali dei tre indici nei tre step futuri |
| `target_mask` | Maschera dei target mancanti |
| `future_target_positions` | Posizioni temporali dei target nella sequenza futura |
| `target_timestamps` | Timestamp delle osservazioni target |
| `climate` | Classe climatica di Köppen |

Le maschere originali usano `True` per i valori mancanti e `False` per i valori disponibili. `future_target_positions` allinea i target alle date nella sequenza futura. La disponibilità di tali posizioni al momento della previsione va verificata nel protocollo di inferenza finale.

## Preprocessing

### Normalizzazione

`ArcsinhScaler` usa le statistiche di `DATA/train_multi_output_clear_sky/scaler.json`. Le covariate e i target usano rispettivamente le coppie `mean` / `var` e `target_mean` / `target_var`. La trasformazione e la sua inversa sono

$$
y = \operatorname{arcsinh}\!\left(\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\right),
\qquad
x = \sinh(y)\sqrt{\sigma^2+\epsilon}+\mu,
\qquad \epsilon = 10^{-5}.
$$

### Valori mancanti del train

Il train comprende 1.455.984 timestep storici nelle finestre. Circa il **78,43%** dei valori di ciascun indice storico è NaN perché la serie giornaliera contiene soltanto tre osservazioni satellitari *clear-sky* per finestra.

| Feature | NaN storici | % sulla feature |
|---------|------------:|----------------:|
| `wind` | 0 | 0,00% |
| `humidity` | 106.469 | 7,31% |
| `solar_irradiance` | 64.210 | 4,41% |
| `rainfall` | 60.128 | 4,13% |
| `pressure` | 147.816 | 10,15% |
| `avg_temperature` | 25.457 | 1,75% |
| `min_temperature` | 60.113 | 4,13% |
| `max_temperature` | 60.113 | 4,13% |
| **Totale** | **524.306** | **4,50% sulle otto feature** |

I conteggi includono tutte le occorrenze nelle finestre, anche quando più finestre possono riferirsi alla stessa misurazione. Nei dati storici del train i NaN delle otto variabili coincidono con le relative maschere. Per alcune aree risultano mancanti intere colonne meteorologiche.

### Weather masking

`preprocess_weather_covariates(raw, missing_mask, scaler)` accetta sequenze di 18 o 21 colonne con la relativa maschera. Legge le prime 18 colonne e restituisce due array `float32`, con 18 covariate normalizzate e otto flag meteorologici nell'ordine originale.

Per ogni feature meteo, il flag vale **1** se il valore è finito e disponibile secondo la maschera originale. Il valore osservato viene trasformato con lo scaler Arcsinh del train. Il flag vale **0** per valori marcati come mancanti o non finiti e il canale normalizzato riceve zero. Lo zero è un placeholder nello spazio normalizzato. La stessa coordinata può valere zero per una misura pari alla media del train, che conserva il flag 1.

| Valore originale con media 20 °C e deviazione standard 5 °C | Valore normalizzato | Flag |
|------------------------------------------------------------|--------------------:|-----:|
| 23 °C | 0,57 | 1 |
| 20 °C | 0,00 | 1 |
| NaN | 0,00 | 0 |

La procedura conserva anche i valori finiti fisicamente anomali, come temperature in ordine incoerente o precipitazioni negative introdotte dal rumore. La verifica della coerenza fisica richiede un controllo di qualità separato. Il trattamento dei buchi meteorologici consiste nel solo zero filling normalizzato con maschere individuali.

`HistoryImputationDataset` concatena le otto maschere al vettore storico. `VegetationDataset` usa la stessa funzione su `history` e su `future` e restituisce `history_weather_mask` e `future_weather_mask`, anche nei batch. I tensori `history` e `future_weather` hanno rispettivamente 21 e 18 canali. Il loader usa le covariate perturbate di `future` e `future_mask` direttamente, lasciando `future_noise` fuori dal preprocessing.

### Feature temporali e clima

`delta_t` misura i giorni dall'ultima osservazione di vegetazione disponibile all'imputer. `orbital_phase` vale `(t % 5) / 5` e `missed_revisits` vale `floor(delta_t / 5)`. Queste feature rappresentano la cadenza nominale di cinque giorni. Ai timestep osservati e prima della prima osservazione `delta_t` e `missed_revisits` valgono zero.

`encode_climate()` assegna il token `<UNKNOWN>` alle classi sconosciute o assenti.

| ID | Classe | Descrizione |
|----|--------|-------------|
| 0 | `<UNKNOWN>` | Classe sconosciuta o non disponibile |
| 1 | BSk | Clima semiarido freddo |
| 2 | Cfa | Clima subtropicale umido |
| 3 | Cfb | Clima oceanico temperato |
| 4 | Csa | Clima mediterraneo con estate calda |
| 5 | Csb | Clima mediterraneo con estate temperata |
| 6 | Dfa | Clima continentale con estate calda |
| 7 | Dfb | Clima continentale con estate temperata |
| 8 | Dfc | Clima subartico |

L'embedding di otto dimensioni viene concatenato a ogni timestep. Durante il training, il climate dropout sostituisce il token climatico con `<UNKNOWN>` con probabilità 0,35. La probabilità è zero in validation.

## Bi-GRU di imputazione storica

`HistoryImputationDataset` costruisce le feature e le maschere a partire da `history`, `history_mask` e `climate`. Riceve soltanto dati storici. La collate function `collate_history_batch` aggiunge padding a destra e restituisce una `padding_mask` con `True` sui timestep reali.

| Canali a partire da 0 | Contenuto | Dimensione |
|----------------------|-----------|-----------:|
| 0–17 | Meteo, nubi e calendario | 18 |
| 18–20 | NDVI, GNDVI ed EVI normalizzati | 3 |
| 21–23 | Maschere delle osservazioni degli indici | 3 |
| 24–26 | Feature temporali e orbitali | 3 |
| 27–34 | Maschere meteorologiche | 8 |
| | **Totale input storico** | **35** |
| 35–42 | Climate embedding | 8 |
| | **Totale dopo la concatenazione** | **43** |

I primi 27 canali mantengono l'ordine della baseline. `BiGRUHistoryImputer` proietta i 43 canali in uno spazio a 64 dimensioni con una sequenza Linear, LayerNorm, ReLU e dropout. Seguono due layer GRU bidirezionali e una testa di ricostruzione. Per un batch di forma `(B, L, 35)`, la Bi-GRU produce uno stato `(B, L, 128)` e la testa restituisce `(B, L, 3)`. `impute_and_blend()` mantiene gli indici osservati e sostituisce i valori mancanti con le ricostruzioni del modello.

La factory di interpolazione offre strategie alternative per gli indici storici, tra cui lineare, forward fill, backward fill, nearest, spline, media, mediana e costante. La strategia `model` carica la stessa Bi-GRU e, in assenza di checkpoint o scaler, usa il fallback lineare. `VegetationDataset` applica di default la strategia lineare solo agli indici. L'addestramento dell'imputer usa gli indici sparsi con mascheramento auto-supervisionato.

### Training e validation interna

Nello split ufficiale train, i primi 70 tile nell'ordinamento dei file costituiscono il fitting set e i successivi 14 la validation interna. Nel fitting viene nascosta un'acquisizione disponibile quando ce ne sono almeno due. I tre valori nascosti per quella data sono il riferimento per la Smooth L1 loss. La regolarizzazione temporale penalizza la differenza quadratica tra predizioni consecutive, limitatamente alle coppie di timestep reali.

| Parametro | Valore |
|-----------|-------:|
| `seed` | 42 |
| `num_train_tiles` | 70 |
| `num_val_tiles` | 14 |
| `epochs` | 10 |
| `batch_size` | 128 |
| `learning_rate` | 0,001 |
| `weight_decay` | 0,0001 |
| `smoothness_weight` | 0,05 |
| `climate_dropout_p` | 0,35 |
| `input_dim` | 35 |
| `climate_dim` | 8 |
| `hidden_dim` | 64 |
| `num_layers` | 2 |
| `output_dim` | 3 |
| `dropout` | 0,1 |

Il training usa AdamW, gradient clipping a 1,0 e lo scheduler ReduceLROnPlateau con fattore 0,5 e patience 1. Usa CUDA se disponibile, altrimenti CPU. Il checkpoint con la loss di validation più bassa viene salvato in `checkpoints/history_imputer_weather_mask_best.pt`.

### Limitazioni note

- La loss di validation interna valuta la ricostruzione di osservazioni visibili anche in input.
- Lo scaler distribuito è stimato su tutto il train ufficiale, compresi i tile della validation interna.
- La padding mask azzera le predizioni sui timestep di padding. La ricorrenza backward può comunque risentire del padding.
- I checkpoint della baseline a 27 canali, incluso `history_imputer_best.pt`, sono incompatibili con la nuova proiezione di input.
- Le incoerenze delle temperature e le anomalie fisiche del meteo perturbato richiedono un controllo di qualità dedicato.
- Il forecaster probabilistico e la valutazione finale devono ancora essere implementati.

## Uso

Per clonare il repository e creare un ambiente virtuale

~~~bash
git clone https://github.com/Gab2112/DeepLearning2026.git
cd DeepLearning2026
python -m venv .venv
~~~

Per attivare l'ambiente usare `.\.venv\Scripts\Activate.ps1` in PowerShell oppure `source .venv/bin/activate` su Bash.

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tests/test_interpolation.py
~~~

Per le analisi preliminari, le funzioni `inspect_sequence_dynamics()`, `inspect_targets_distribution()` e `inspect_scaler_and_transforms()` in `src/data/inspect_data.py` leggono il solo train. Lo script completo include anche l'inventario degli split ufficiali.

Per avviare il training dell'imputer con i parametri di `configs/config.yaml`

~~~bash
python -m src.train_imputer
~~~

Con il nuovo checkpoint disponibile, la CLI dell'imputer offre inferenza su un esempio del train e valutazione leave-one-out sullo stesso esempio.

~~~bash
python src/data/bigru_imputer.py --mode test
python src/data/bigru_imputer.py --mode evaluate
~~~

### Moduli principali

| File | Responsabilità |
|------|----------------|
| `src/data/transforms.py` | Normalizzazione Arcsinh e trasformazione inversa |
| `src/data/weather_preprocessor.py` | Covariate normalizzate e maschere meteorologiche per history e future |
| `src/data/history_dataset.py` | Preparazione e batching dell'input della Bi-GRU |
| `src/data/vegetation_dataset.py` | Loader per forecasting, con sequenze storiche, covariate future e target |
| `src/data/interpolation.py` | Factory per le strategie sugli indici storici |
| `src/data/climate.py` | Codifica climatica |
| `src/models/history_imputer.py` | Architettura Bi-GRU |
| `src/data/bigru_imputer.py` | Caricamento checkpoint, inferenza singola e batch, CLI |
| `src/train_imputer.py` | Training, validation e checkpointing |
| `src/utils/config.py` | Caricamento YAML e risoluzione dei percorsi della sezione `paths` |
| `configs/config.yaml` | Dati, modello, training e specifiche del futuro forecaster |
| `tests/test_weather_preprocessor.py` | Test di preprocessing, layout storico e compatibilità dei checkpoint |
| `tests/test_interpolation.py` | Test delle strategie di interpolazione e integrazione con il dataset |

## Forecasting e valutazione finale

La configurazione specifica i tre step e i quantili del futuro forecaster. La Pinball Loss misura l'errore dei quantili. L'intervallo tra 0.1 e 0.9 ha coverage nominale dell'80%. Il beneficio dell'imputazione e del weather masking per il forecasting resta da misurare.

La valutazione deve riportare RMSE, MAE, WMAPE e MASE sulla mediana 0.5, oltre a Pinball Loss, CRPS e coverage per i quantili. Il protocollo prevede metriche per indice e per split, dopo trasformazione inversa e sui target reali. La baseline per MASE e l'approssimazione del CRPS con tre quantili sono da definire. Gli split ID, OOD-S e OOD-T sono riservati a questa fase. Le metriche sperimentali di forecasting sono in attesa dell'implementazione del modello.

## Autori

- Nicole Bovolenta
- Daniele Cantagallo
- Gabriele Orazio Di Raimondo
