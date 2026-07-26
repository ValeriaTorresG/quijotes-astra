# Power spectra de Quijote por ambiente ASTRA

`compute_power_spectra.py` reproduce el procedimiento de
`power_spec.ipynb` para cada realización:

1. lee las posiciones de halos FoF con `readfof`;
2. calcula el espectro de todos los halos;
3. asigna cada halo a `void`, `sheet`, `filament` o `knot` mediante el
   máximo de `PVOID`, `PSHEET`, `PFILAMENT`, `PKNOT`;
4. calcula los cuatro espectros ambientales;
5. guarda un CSV independiente por muestra.

Los defaults numéricos son los del notebook: caja periódica de `1000 Mpc/h`,
malla `512^3`, `CIC`, eje de línea de visión `x`, un thread y
`0.008 <= k <= 0.5 h/Mpc`.

## Uso

Ejecutar desde la raíz de este repositorio:

```bash
conda activate pylians-x86
```

Todas las realizaciones disponibles de `fiducial`, snapshot 3:

```bash
python pk/compute_power_spectra.py \
  --dataset fiducial \
  --snapnum 3
```

Una selección explícita:

```bash
python pk/compute_power_spectra.py \
  --dataset fiducial \
  --simulation-ids 0 1 10 100 101 102 103 104 105 106 \
  --snapnum 3
```

Las realizaciones disponibles de `Om_m`:

```bash
python pk/compute_power_spectra.py \
  --dataset Om_m \
  --snapnum 3
```

Solo `Om_m`, realización 0:

```bash
python pk/compute_power_spectra.py \
  --dataset Om_m \
  --simulation-id 0 \
  --snapnum 3
```

Todas las familias que tengan la estructura
`data/<familia>/<realizacion>/groups_003/group_tab_003.0`:

```bash
python pk/compute_power_spectra.py \
  --datasets all \
  --snapnum 3
```

Las rutas completas también se pueden pasar explícitamente:

```bash
python pk/compute_power_spectra.py \
  --dataset fiducial \
  --snapnum 3 \
  --data-root "$HOME/Desktop/Quijotes/data" \
  --astra-root "$HOME/Desktop/Quijotes/data/astra" \
  --output-root "$HOME/Desktop/Quijotes/data/pk"
```

Antes de una corrida grande se pueden revisar todas las rutas sin calcular:

```bash
python pk/compute_power_spectra.py \
  --datasets all \
  --snapnum 3 \
  --dry-run
```

## Productos

Para la realización 0 de `fiducial`, los nombres son:

```text
data/pk/matter/fiducial_sim000_snap003_pk.csv
data/pk/env/fiducial_sim000_snap003_void_pk.csv
data/pk/env/fiducial_sim000_snap003_sheet_pk.csv
data/pk/env/fiducial_sim000_snap003_filament_pk.csv
data/pk/env/fiducial_sim000_snap003_knot_pk.csv
```

Cada CSV contiene `k`, `P0`, `P2`, `P4`, número de modos, el estimador de error
del monopolo usado en el notebook, shot noise y el monopolo con shot noise
restado, además de los parámetros y unidades usados. La columna
`sigma_Pk0_notebook_Mpc3_h3` conserva literalmente la fórmula del notebook,
`P0*sqrt(2/Nmodes)`.

También se guardan hashes SHA-256 del catálogo FoF y del FITS de probabilidades,
el número de iteraciones ASTRA, semilla, umbrales y geometría periódica. Antes
de omitir un archivo existente, el script comprueba su esquema, integridad,
configuración y procedencia. Si no coincide con la corrida solicitada, termina
con un mensaje que pide usar `--overwrite`; así no se mezcla, por ejemplo, un
CSV de una prueba `32^3` con una corrida científica `512^3`.

El directorio `matter/` conserva el nombre solicitado, pero su muestra
`sample=all` corresponde a **todos los halos FoF reales**, igual que `All` en
el notebook. No es el campo de partículas de materia oscura.

## Productos ASTRA incompletos

Los espectros ambientales requieren el archivo final
`*_probability_iterdata.fits.gz`. Los chunks de pares o clasificaciones no son
suficientes. Sin ese archivo, la simulación se reporta como fallida; el
espectro de todos los halos sí puede haberse escrito antes.

Para calcular todos los espectros de halos disponibles y omitir temporalmente
los ambientes que todavía no tengan probabilidades finales:

```bash
python pk/compute_power_spectra.py \
  --datasets all \
  --snapnum 3 \
  --allow-missing-probabilities
```

Para calcular únicamente todos los halos, sin depender de ASTRA:

```bash
python pk/compute_power_spectra.py \
  --datasets all \
  --snapnum 3 \
  --matter-only
```

El número de iteraciones ASTRA no se pasa de nuevo a este script: se lee de la
cabecera del FITS de probabilidades y se registra en
`n_astra_iterations`. Esto funciona tanto para 45 como para 100 iteraciones.

El script también rechaza un `--kmax` mayor que la frecuencia de Nyquist de la
malla elegida.
