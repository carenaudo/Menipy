# Probe Liquid Reference & Database

This document catalogs the standard test probe liquids implemented in Menipy's
Surface Free Energy (SFE) library (`menipy.common.liquid_db`). Data and partitioning
comply with ISO 19403-2 and DIN 55660-2 international standards.

## Reference Liquid Table

All surface tension values are given in $\mathrm{mN/m}$ (equivalent to $\mathrm{mJ/m^2}$).

| Liquid | Formula | CAS | Temp (°C) | $\gamma_L$ | $\gamma_L^d$ | $\gamma_L^p$ | OWRK $x$ | Primary Reference | DOI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Water** | $\mathrm{H_2O}$ | 7732-18-5 | 20.0 | 72.8 | 21.8 | 51.0 | 1.530 | Ström et al. (1987) | [10.1016/0021-9797(87)90043-2](https://doi.org/10.1016/0021-9797(87)90043-2) |
| **Water** | $\mathrm{H_2O}$ | 7732-18-5 | 25.0 | 72.0 | 21.8 | 50.2 | 1.517 | Jasper (1972); Good & van Oss (1991) | [10.1021/je60054a027](https://doi.org/10.1021/je60054a027) |
| **Diiodomethane** | $\mathrm{CH_2I_2}$ | 75-11-6 | 20.0 | 50.8 | 50.8 | 0.0 | 0.000 | Owens & Wendt (1969); Ström et al. (1987) | [10.1002/app.1969.070130815](https://doi.org/10.1002/app.1969.070130815) |
| **Diiodomethane** | $\mathrm{CH_2I_2}$ | 75-11-6 | 25.0 | 50.0 | 50.0 | 0.0 | 0.000 | Fowkes (1964); Good & van Oss (1991) | [10.1021/ie50571a036](https://doi.org/10.1021/ie50571a036) |
| **Glycerol** | $\mathrm{C_3H_8O_3}$ | 56-81-5 | 20.0 | 64.0 | 34.0 | 30.0 | 0.939 | Ström et al. (1987); Busscher et al. (1984) | [10.1016/0021-9797(87)90043-2](https://doi.org/10.1016/0021-9797(87)90043-2) |
| **Glycerol** | $\mathrm{C_3H_8O_3}$ | 56-81-5 | 25.0 | 63.4 | 37.0 | 26.4 | 0.845 | van Oss et al. (1988) | [10.1016/0009-2614(88)85051-4](https://doi.org/10.1016/0009-2614(88)85051-4) |
| **Ethylene glycol** | $\mathrm{C_2H_6O_2}$ | 107-21-1 | 20.0 | 48.0 | 29.0 | 19.0 | 0.810 | Ström et al. (1987); Kaelble (1970) | [10.1016/0021-9797(87)90043-2](https://doi.org/10.1016/0021-9797(87)90043-2) |
| **Ethylene glycol** | $\mathrm{C_2H_6O_2}$ | 107-21-1 | 25.0 | 47.7 | 29.3 | 18.4 | 0.792 | van Oss et al. (1988) | [10.1016/0009-2614(88)85051-4](https://doi.org/10.1016/0009-2614(88)85051-4) |
| **Formamide** | $\mathrm{CH_3NO}$ | 75-12-7 | 20.0 | 58.0 | 39.5 | 18.5 | 0.684 | Ström et al. (1987); Owens & Wendt (1969) | [10.1016/0021-9797(87)90043-2](https://doi.org/10.1016/0021-9797(87)90043-2) |
| **DMSO** | $\mathrm{C_2H_6OS}$ | 67-68-5 | 20.0 | 44.0 | 36.0 | 8.0 | 0.471 | Janczuk et al. (1993); Wu (1982) | [10.1016/0021-9797(93)90386-M](https://doi.org/10.1016/0021-9797(93)90386-M) |
| **1-Bromonaphthalene** | $\mathrm{C_{10}H_7Br}$ | 90-11-9 | 20.0 | 44.4 | 44.4 | 0.0 | 0.000 | Fowkes (1964); Ström et al. (1987) | [10.1021/ie50571a036](https://doi.org/10.1021/ie50571a036) |
| **1-Bromonaphthalene** | $\mathrm{C_{10}H_7Br}$ | 90-11-9 | 25.0 | 44.6 | 44.6 | 0.0 | 0.000 | Good & van Oss (1991) | [10.1007/978-1-4615-3106-2_7](https://doi.org/10.1007/978-1-4615-3106-2_7) |
| **Hexadecane** | $\mathrm{C_{16}H_{34}}$ | 544-76-3 | 20.0 | 27.5 | 27.5 | 0.0 | 0.000 | Jasper (1972); Fowkes (1964) | [10.1021/je60054a027](https://doi.org/10.1021/je60054a027) |
| **Thiodiglycol** | $\mathrm{C_4H_{10}O_2S}$ | 111-48-8 | 20.0 | 54.0 | 14.8 | 39.2 | 1.626 | Berger (1991); Janczuk et al. (1993) | [10.1016/0021-9797(93)90386-M](https://doi.org/10.1016/0021-9797(93)90386-M) |
| **Tricresyl phosphate** | $\mathrm{C_{21}H_{21}O_4P}$ | 1330-78-5 | 20.0 | 40.9 | 39.2 | 1.7 | 0.208 | Panzer (1973); Wu (1982) | [10.1016/0021-9797(73)90004-0](https://doi.org/10.1016/0021-9797(73)90004-0) |
| **Benzyl alcohol** | $\mathrm{C_7H_8O}$ | 100-51-6 | 20.0 | 39.0 | 29.0 | 10.0 | 0.587 | Kaelble (1970); Panzer (1973) | [10.1016/S0021-9797(70)80035-9](https://doi.org/10.1016/S0021-9797(70)80035-9) |

---

## Experimental Handling and Recommendations

1. **Water**: Primary polar standard liquid. Must be high-purity (Type 1 Milli-Q, resistivity $18.2\,\mathrm{M\Omega\cdot cm}$, TOC $< 5\,\mathrm{ppb}$).
2. **Diiodomethane**: Primary dispersive standard. Protect from light exposure (keep in dark glass with copper wire stabiliser). If liquid darkens to brown/red, iodine decomposition has occurred and calibration values will be inaccurate.
3. **Glycerol & Ethylene Glycol**: Viscous, hygroscopic polar probe liquids. Seal bottles immediately after use to prevent ambient moisture absorption which alters both density and polar surface tension.
4. **Formamide**: High polar component but teratogenic; dispense in a fume hood. Can dissolve or craze acrylic (PMMA), polycarbonate, and certain polyamides.
5. **Hexadecane**: Non-polar alkane. Useful for low-energy surfaces (e.g. PTFE, fluoropolymer coatings) where higher surface tension probe liquids do not wet sufficiently.

---

## Bibliography

- **Berger, E. J.** (1991). *Adhesion Information from Contact Angle Measurement*, in *Adhesion and Adsorption of Polymers*, ed. L.-H. Lee, Plenum Press, pp. 297–315.
- **Busscher, H. J., Van Pelt, A. W. J., De Jong, H. P., & Arends, J.** (1984). *Effect of spreading pressure on surface free energy determination by means of contact angle measurements*. **J. Colloid Interface Sci.**, 99(2), 342–347. DOI: 10.1016/0021-9797(84)90123-5.
- **Fowkes, F. M.** (1964). *Attractive forces at interfaces*. **Ind. Eng. Chem.**, 56(12), 40–52. DOI: 10.1021/ie50571a036.
- **Good, R. J., & van Oss, C. J.** (1991). *The Modern Theory of Contact Angles and the Hydrogen Bond Components of Surface Energies*, in *Modern Approaches to Wettability*, ed. M. E. Schrader & G. I. Loeb, Springer, pp. 1–27. DOI: 10.1007/978-1-4615-3106-2_1.
- **Janczuk, B., Bialopiotrowicz, T., & Zdziennicka, A.** (1993). *Components of the surface tension of liquid mixtures and their adsorption at water-air interfaces*. **J. Colloid Interface Sci.**, 159(2), 421–428. DOI: 10.1016/0021-9797(93)90386-M.
- **Jasper, J. J.** (1972). *The surface tension of pure liquid compounds*. **J. Phys. Chem. Ref. Data**, 1(4), 841–1010. DOI: 10.1063/1.3253106.
- **Kaelble, D. H.** (1970). *Dispersion-polar surface tension properties of organic solids*. **J. Adhes.**, 2(2), 66–81. DOI: 10.1080/0021846708544582.
- **Owens, D. K., & Wendt, R. C.** (1969). *Estimation of the surface free energy of polymers*. **J. Appl. Polym. Sci.**, 13(8), 1741–1747. DOI: 10.1002/app.1969.070130815.
- **Panzer, J.** (1973). *Components of solid surface free energy from wetting measurements*. **J. Colloid Interface Sci.**, 44(1), 142–161. DOI: 10.1016/0021-9797(73)90004-0.
- **Ström, G., Fredriksson, M., & Stenius, P.** (1987). *Contact angles, work of adhesion, and interfacial tensions at a dissolving cellulose surface*. **J. Colloid Interface Sci.**, 119(2), 352–361. DOI: 10.1016/0021-9797(87)90043-2.
- **van Oss, C. J., Chaudhury, M. K., & Good, R. J.** (1988). *Interfacial Lifshitz-van der Waals and polar interactions in macroscopic systems*. **Chem. Rev.**, 88(6), 927–941. DOI: 10.1021/cr00088a006.
- **Wu, S.** (1982). *Polymer Interface and Adhesion*, Marcel Dekker, New York.
