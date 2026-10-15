# Hippo002: MRT–XPCT-Tabellen für die Korrelation

Eine Zeile ist ein MRT-Voxel, dessen Mittelpunkt oder Abtastung in der XPCT-Maske liegt. Karten auf demselben Voxelgitter stehen in derselben Datei. Alle Zeilen sind enthalten, auch die unbrauchbaren. Für die Korrelation `coord_usable == 1` verwenden.

Die Dateinamen nennen die Stanze und das Gitter, der Ordner den MRT-Raum.

## Stanzen

| Dateiname | Probe | Tomogramme in der Zellsegmentierung |
|---|---|---|
| `P2_` | H2P2 | 0001, 0008, 0015, 0022, 0029, 0036 |
| `P5_` | H2P5 | 0022, 0029 |
| `P7_` | H2P7 | 0001, 0002, 0003, 0004 |

## Ordner und Kontraste

`HH_exvivo`, Hamburg ex vivo. MPM, MD, MK und MW liegen auf 0,3 mm. FA, R2 und MWI haben eigene Gitter.

| Datei | Spalten |
|---|---|
| `P*_mpm_0p3mm_in_tomos.csv` | A, R1, R2s, A_denoised, R1_denoised, R2s_denoised, MD, MK, MW |
| `P*_dki_0p8mm_in_tomos.csv` | FA |
| `P*_r2_0p5mm_in_tomos.csv` | R2 |
| `P*_mwi_0p5mm_in_tomos.csv` | MWF, R1_freewater, R2s_intraaxonal, R2s_myelinwater, T1_freewater |

`HH_invivo`, Hamburg in vivo. MPM und DKI teilen sich jeweils ein Gitter.

| Datei | Spalten |
|---|---|
| `P*_mpm_1mm_in_tomos.csv` | MTsat, PD, R1, R2s |
| `P*_dki_1p7mm_in_tomos.csv` | FA, MD, MK, MW |

`Portugal_R2s`: nur `R2s`, 0,05 mm.
`Portugal_DKI`: FA, MD, MK, 0,1 mm.
`Leipzig`: A, R1, R2s, T1, 0,3 mm.

Wo Original und denoised nebeneinander stehen, sind das zwei Auswertungen desselben Kontrasts.

## Welche Zeilen korrelieren

`coord_usable` ist 1, wenn der Voxelmittelpunkt in mindestens einem Originaltomogramm der Zellsegmentierung liegt und dort in den brauchbaren Bereich fällt. Sonst 0. Diese Zeilen nicht löschen, aber für die Korrelation weglassen.

`tomogram_usable` nennt die Tomogramme, in denen der Mittelpunkt brauchbar ist. `0001+0008` ist ein Überlapp. Histologie dann aus einem der genannten Tomogramme lesen. `tomogram` listet alle berührten Tomogramme, auch die unbrauchbaren.

`corners_usable` ist strenger: Mittelpunkt und alle acht Ecken des MRT-Voxels müssen im selben Tomogramm brauchbar sein. Bei 0,8 mm und gröber ist das oft 0, weil eine Ecke aus dem gültigen Bereich ragt. Für die voxelweise Korrelation reicht `coord_usable`.

Leere Tomogrammspalten heißen: dieser Mittelpunkt berührt die Kachel nicht.

## Brauchbarer Bereich im Originaltomogramm

Index des nächsten Voxels, Achsen i, j, k = x, y, z. Das Tomogramm ist 2560 × 2560 × 2156 bei 650 nm.

H2P5: brauchbar, sobald der Punkt im Volumen liegt.

H2P7, alle vier Tomogramme: i von 744 bis 1464 und j von 933 bis 1653, jeweils einschließlich. Das ist absichtlich nur ein kleiner Teil des Gesichtsfelds, etwa 9 Prozent der Voxel.

H2P2:

| Tomogramm | bleibt | fällt weg |
|---|---|---|
| 0001 | k ab 589 | 0 bis 588 |
| 0008 | alles im Volumen | |
| 0015 | 0 bis 1366 | ab 1367 |
| 0022 | ab 736 | 0 bis 735 |
| 0029 | alles im Volumen | |
| 0036 | 0 bis 1841 | ab 1842 |

## Koordinaten

`mri_i`, `mri_j`, `mri_k` sind der MRT-Voxel, 0-basiert, dieselbe Achsenreihenfolge wie im NIfTI. In Matlab liegt dasselbe Voxel bei Index + 1.

`xpct_i`, `xpct_j`, `xpct_k` sind der Mittelpunkt im 2,6-µm-Stitch. Index 0 ist der Mittelpunkt des ersten Voxels. `xpct_*_650nm` ist derselbe Punkt im ungebinnnten Stitch (`xpct_2p6um * 4`). `xpct_*_nearest` ist `floor(Index + 0.5)`.

`c000` bis `c111` sind die acht Ecken. Die drei Ziffern stehen für i, j, k. 0 ist Index minus 0,5, 1 ist Index plus 0,5. `c000` ist (i−0,5, j−0,5, k−0,5), `c111` ist (i+0,5, j+0,5, k+0,5).

`tomoXXXX_i_650nm`, `_j_650nm`, `_k_650nm` sind derselbe Mittelpunkt im Originaltomogramm der Zellsegmentierung. `tomoXXXX_*_nearest` ist der Voxel im OME-Zarr. Dort ist die Achsenreihenfolge z, y, x, also k, j, i. `tomoXXXX_in_volume` ist 1, wenn dieser Voxel im Tomogramm liegt. `tomoXXXX_in_circle` ist 1, wenn der Punkt in dem Kreis liegt, der in den Stitch übernommen wurde. `tomoXXXX_usable` ist 1, wenn der Mittelpunkt im Volumen liegt und die Regel oben erfüllt. Dieselben Spalten mit Vorsilbe `cABC_` gelten für die Ecken.

`xpct_mean` ist die mittlere XPCT-Intensität der Abtastpunkte in der Maske, `xpct_samples` ihre Anzahl. `coverage_percent` ist der Anteil der Abtastpunkte in der Maske. 100 heißt, das ganze MRT-Voxel liegt in der Maske. Abgetastet wird in Schritten von höchstens 8 µm.
