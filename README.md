# IceCube 10 Yıllık Veriyle Nötrino Nokta Kaynak Analizi

IceCube'un halka açık 10 yıllık iz (track) verisiyle (2008–2018, ~1.1 milyon olay) gökyüzünde nötrino kaynağı arayan, sıfırdan yazılmış bir **unbinned likelihood** analizi. IceCube'un 2020 PRL sonuçları (NGC 1068) ve 2018 TXS 0506+056 patlaması bağımsız olarak yeniden üretildi.

**Yazar:** Musab Bera Mansur Oluğ — AYBÜ Matematik, 1. sınıf

**Proje sitesi:** https://KULLANICI_ADIN.github.io/icecube-notrino-analizi/ (etkileşimli açıklamalarla)  
**Rapor:** `IceCube 10 Yıllık Veriyle Nötrino Nokta Kaynak Analizi.pdf`

## Öne çıkan sonuçlar

| Analiz | Bu proje | IceCube yayını |
|---|---|---|
| NGC 1068 nₛ / γ | 49.6 / 3.2 | 50.4 / 3.2 (PRL 2020) |
| NGC 1068 yerel anlamlılık | 3.94σ | ≈4.1σ |
| Kuzey en sıcak nokta | 40.87°, −0.5° | 40.9°, −0.3° |
| En sıcak nokta post-trial | ≈1.9σ | 1.29σ |
| TXS 0506+056 2014–15 patlaması | 4.1σ (kutu), γ=2.3 | 3.5σ, γ=2.1 (Science 2018) |
| Güney gökyüzü / Gökada düzlemi | sıfır sonuç | sıfır sonuç (iz olayları) |

Tam tablo: `sonuc_karsilastirma.csv`

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Veri:** IceCube 10 yıllık nokta kaynak verisi (Harvard Dataverse, doi:10.7910/DVN/VKL316) → `data/dataverse_files/` altına açın (`events/`, `irfs/`, `uptime/`).
Fermi-LAT 4FGL-DR4 kataloğu (`gll_psc_v35.fit`) → `data/` (Aşama 10 notebook'u otomatik indirmeyi dener).

## Klasör yapısı

```
src/icecube_ps.py     tüm analiz kodu (likelihood, tarama, sezonlu model, paralel çalıştırma)
notebooks/            aşama aşama notebook'lar
results/              grafikler ve tablolar (csv/png)
docs/                 proje sitesi (GitHub Pages)
data/                 veri (repoda yok)
```

## Çalıştırma sırası

| # | Notebook | Ne yapıyor | Süre* |
|---|---|---|---|
| 01 | veri_kesfi | Veriye ilk bakış, dağılımlar | dk |
| 02 | kaynak_sayim_montecarlo | Basit sayım + Monte Carlo | dk |
| 03 | likelihood_anlamlilik | Unbinned likelihood, TS, p-değeri | dk |
| 05 | hassasiyet_ust_sinir | Hassasiyet, 5σ keşif, üst sınır, akı | ~10 dk |
| 06 | stacking | 4 kaynağın birlikte testi | ~5 dk |
| 07 | gok_taramasi | Kuzey gökyüzü taraması | ~20 dk |
| 08 | txs_patlama | Zaman bağımlı patlama taraması | ~10 dk |
| 09 | enerji | Enerji terimi ve γ uyumu | ~10 dk |
| 10 | fermi_blazarlar | 100 Fermi blazarı (stacking + tekil) | ~20 dk |
| 11 | performans | Paralel çalıştırma, post-trial | ~30 dk |
| 12 | sezonlu_model | Sezon başına IRF'li tam model | ~15 dk |
| 13 | enerjili_tarama | Enerjili tam gök taraması + post-trial | saatler |
| 14 | txs_onceden_pencere | Önceden seçilmiş zaman penceresi | ~10 dk |
| 15 | saglamlik | Sistematik testler | ~30 dk |
| 16 | guney_tarama | Güney gökyüzü taraması | saatler |
| 17 | gokada_duzlemi | Gökada düzlemi şablon analizi | ~20 dk |

\*Ryzen 7600X, 12 iş parçacığı.

## Yöntem (kısaca)

- **Likelihood:** L(nₛ, γ) = Π [ (nₛ/N)·Sᵢ + (1 − nₛ/N)·Bᵢ ]; S = konum (PSF) × enerji olasılığı, B = veriden (RA karıştırmalı) arka plan.
- **Anlamlılık:** TS = 2 ln[L(n̂ₛ)/L(0)]; p-değeri RA-karıştırılmış binlerce sahte gökyüzünden.
- **Bakış-başka-yerde (look-elsewhere):** gerçek ve karışık haritalara birebir aynı tarama uygulanıp doğrudan sayımla post-trial.
- **Akı:** etkin alan × canlı süre ile nₛ → E²Φ (1 TeV).

## Sınırlamalar

Halka açık veri resmi analizden daha kaba IRF'ler içerir; NGC 1068'in anlamlılığı ±0.2° konum değişimine duyarlı (2.8–4.4σ). Gelecek adım: IceTracks-DR2 (2008–2022) verisiyle tekrar.

## Kaynaklar

- IceCube Collab., *PRL* 124, 051103 (2020) — 10 yıllık zaman entegre arama
- IceCube Collab., *Science* 361, 147 (2018) — TXS 0506+056 patlaması
- IceCube Collab., *Science* 378, 538 (2022) — NGC 1068
- IceCube Collab., *Science* 380, 1338 (2023) — Gökada düzlemi
- Braun et al., *Astropart. Phys.* 29, 299 (2008) — unbinned likelihood yöntemi
