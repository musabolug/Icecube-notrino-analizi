"""
icecube_ps: IceCube 10 yıllık açık verisiyle nokta kaynak analizi için ortak araçlar.

Notebook'larda:
    import sys; sys.path.append('../src')
    import icecube_ps as ps
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import UnivariateSpline

SURUM = '15.0'   # notebook'lar doğru sürümün yüklendiğini bununla kontrol eder

KAYNAKLAR = {
    'TXS 0506+056':   (77.358,   5.693),
    'NGC 1068':       (40.670,  -0.013),
    'PKS 1424+240':   (216.752, 23.800),
    'GB6 J1542+6129': (235.737, 61.499),
}

# --------------------------------------------------------------------------
# Veri
# --------------------------------------------------------------------------
SUTUNLAR = ['mjd', 'logE', 'angErr', 'ra', 'dec', 'az', 'zen']


def veri_yukle(data_dir='../data', onbellek=True):
    """Tüm sezonları tek tabloda topla. İlk seferden sonra hızlı yükleme için
    data/olaylar.pkl dosyasına önbellek yazar."""
    data_dir = Path(data_dir)
    pkl = data_dir / 'olaylar.pkl'
    if onbellek and pkl.exists():
        return pd.read_pickle(pkl)
    dfs = []
    for f in sorted(data_dir.rglob('events/*.csv')):
        d = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=SUTUNLAR)
        d['season'] = f.stem.replace('_exp', '')
        dfs.append(d)
    ev = pd.concat(dfs, ignore_index=True)
    if onbellek:
        ev.to_pickle(pkl)
    return ev


# --------------------------------------------------------------------------
# Geometri
# --------------------------------------------------------------------------
def acisal_uzaklik(ra1, dec1, ra2, dec2):
    """İki gök noktası arasındaki açı. Girdi ve çıktı derece."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cos_psi = (np.sin(dec1) * np.sin(dec2)
               + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(cos_psi, -1, 1)))


# --------------------------------------------------------------------------
# Arka plan olasılık yoğunluğu  B(δ)  [1/steradyan]
# --------------------------------------------------------------------------
class ArkaPlan:
    """
    Arka planın yönsel dağılımı, verinin kendisinden.
    RA boyunca düzgün (Dünya'nın dönüşü), sin(Dec) boyunca histogramdan:
        B(δ) = P(sin δ) / 2π
    Histogram pürüzlü olduğu için log-yoğunluğa yumuşak bir spline uyduruyoruz.
    """

    def __init__(self, dec_deg, n_kutu=100, yumusaklik=None):
        sin_dec = np.sin(np.radians(dec_deg))
        sayim, kenar = np.histogram(sin_dec, bins=n_kutu, range=(-1, 1))
        genislik = kenar[1] - kenar[0]
        yogunluk = sayim / (sayim.sum() * genislik)          # ∫ P d(sinδ) = 1
        merkez = 0.5 * (kenar[1:] + kenar[:-1])
        # Poisson hatasına göre ağırlıklı spline: çok olay olan kutuya daha çok güven
        agirlik = np.sqrt(np.maximum(sayim, 1))
        if yumusaklik is None:
            yumusaklik = n_kutu
        self._spline = UnivariateSpline(merkez, np.log(np.maximum(yogunluk, 1e-12)),
                                        w=agirlik, s=yumusaklik, k=3)
        self.merkez, self.yogunluk = merkez, yogunluk

    def P(self, sin_dec):
        return np.exp(self._spline(sin_dec))

    def __call__(self, dec_deg):
        """B(δ): steradyan başına arka plan olasılık yoğunluğu."""
        return self.P(np.sin(np.radians(dec_deg))) / (2 * np.pi)


# --------------------------------------------------------------------------
# Sinyal olasılık yoğunluğu  S_i  [1/steradyan]
# --------------------------------------------------------------------------
def sinyal_pdf(psi_deg, sigma_deg):
    """Kaynak etrafında 2B Gauss: kaynağa yakın ve kesin olaylar yüksek değer alır."""
    psi, sig = np.radians(psi_deg), np.radians(sigma_deg)
    return np.exp(-psi**2 / (2 * sig**2)) / (2 * np.pi * sig**2)


# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------
def ns_fit_scipy(sb, N):
    """S/B dizisi ve toplam olay sayısı N verildiğinde nₛ'i maksimize et.
    Uzak olaylar (S≈0) sadece (N − len(sb))·log(1 − ns/N) terimiyle katkı verir."""
    if len(sb) == 0 or np.sum(sb) - N <= 0:      # ns=0'da eğim ≤ 0 → fazlalık yok
        return 0.0, 0.0
    def eksi_ll(ns):
        return -(np.sum(np.log1p(ns / N * (sb - 1))) + (N - len(sb)) * np.log1p(-ns / N))
    sonuc = optimize.minimize_scalar(eksi_ll, bounds=(0, min(len(sb), 1000)),
                                     method='bounded', options={'xatol': 1e-3})
    return sonuc.x, max(0.0, -2 * eksi_ll(sonuc.x))


def ns_fit_newton(sb, N, tol=1e-6, max_iter=60):
    """
    Aynı fit, ama genel amaçlı optimize edici yerine likelihood'un türevinin
    (skor fonksiyonu) kökünü Newton yöntemiyle buluyoruz:
        f(nₛ) = Σ (sbᵢ−1) / (N + nₛ(sbᵢ−1))  −  (N−n) / (N−nₛ)  = 0
    f azalan bir fonksiyon; [alt, üst] aralığını koruyarak (gerekirse ikiye bölerek)
    birkaç adımda yakınsar. (Aşama 11)
    """
    m = len(sb)
    if m == 0 or np.sum(sb) - N <= 0:
        return 0.0, 0.0
    x = sb - 1.0
    def ll(ns):
        return np.sum(np.log1p(ns / N * x)) + (N - m) * np.log1p(-ns / N)
    alt, ust = 0.0, float(min(m, 1000))
    def f_ve_turev(ns):
        q = x / (N + ns * x)
        return np.sum(q) - (N - m) / (N - ns), -np.sum(q * q) - (N - m) / (N - ns) ** 2
    f_ust, _ = f_ve_turev(ust)
    if f_ust > 0:                      # kök sınırın ötesinde → sınırda dur
        return ust, max(0.0, 2 * ll(ust))
    ns = 0.0
    for _ in range(max_iter):
        f, df = f_ve_turev(ns)
        if f > 0:
            alt = ns
        else:
            ust = ns
        yeni = ns - f / df
        if not (alt < yeni < ust):     # Newton adımı aralıktan çıkarsa ikiye böl
            yeni = 0.5 * (alt + ust)
        if abs(yeni - ns) < tol:
            ns = yeni
            break
        ns = yeni
    return ns, max(0.0, 2 * ll(ns))


ns_fit = ns_fit_newton      # Aşama 11'den itibaren varsayılan (ns_fit_scipy ile aynı sonuç, daha hızlı)


class NoktaKaynakLikelihood:
    """
    log L(ns) = Σ_i log[ ns/N · S_i + (1 − ns/N) · B_i ]

    Hız hilesi: kaynaktan uzaktaki olaylarda S_i ≈ 0, katkıları log(1 − ns/N).
    Bu yüzden sadece PENCERE derece içindeki olayları tek tek hesaplıyoruz:

    TS/2 = Σ_yakın log[1 + ns/N (S_i/B_i − 1)] + (N − n_yakın) · log(1 − ns/N)
    """

    PENCERE = 10.0     # derece

    def __init__(self, ev, arka_plan, ra_k, dec_k):
        self.N = len(ev)
        self.ra_k, self.dec_k = ra_k, dec_k
        self.B = arka_plan
        # Dec'i yakın olanlar: karıştırmada da yalnızca bunlar daireye girebilir
        bant = np.abs(ev['dec'].values - dec_k) < self.PENCERE
        self.dec_bant = ev['dec'].values[bant]
        self.ra_bant = ev['ra'].values[bant]
        self.sig_bant = ev['angErr'].values[bant]
        self.B_bant = arka_plan(self.dec_bant)          # Dec değişmediği için bir kere
        # Monte Carlo hızı için sin/cos çarpımlarını önceden hesapla
        d, dk = np.radians(self.dec_bant), np.radians(dec_k)
        self._A = np.sin(d) * np.sin(dk)
        self._C = np.cos(d) * np.cos(dk)

    # ---- yardımcılar ----
    def _psi(self, ra):
        cos_psi = self._A + self._C * np.cos(np.radians(ra - self.ra_k))
        return np.degrees(np.arccos(np.clip(cos_psi, -1, 1)))

    def sb_orani(self, ra=None):
        """Her yakın olay için S_i / B_i. ra verilmezse gerçek RA'lar."""
        ra = self.ra_bant if ra is None else ra
        psi = self._psi(ra)
        yakin = psi < self.PENCERE
        return sinyal_pdf(psi[yakin], self.sig_bant[yakin]) / self.B_bant[yakin]

    def log_lr(self, ns, sb, N=None):
        """log L(ns) − log L(0) = TS/2"""
        N = self.N if N is None else N
        return (np.sum(np.log1p(ns / N * (sb - 1)))
                + (N - len(sb)) * np.log1p(-ns / N))

    def fit_sb(self, sb, N=None):
        """Hazır S/B dizisinden nₛ fit'i. (ns_hat, TS) döndürür."""
        return ns_fit(sb, self.N if N is None else N)

    def fit(self, ra=None):
        """nₛ'i maksimize et. (ns_hat, TS) döndürür."""
        return self.fit_sb(self.sb_orani(ra))

    def profil(self, ns_dizisi, ra=None):
        """−2Δlog L eğrisi (en iyi noktaya göre). Belirsizlik aralığı için."""
        sb = self.sb_orani(ra)
        ll = np.array([self.log_lr(n, sb) for n in ns_dizisi])
        return -2 * (ll - ll.max())

    def arka_plan_denemeleri(self, n_deneme, rng):
        """RA'ları karıştırıp TS dağılımını üret (H₀: sadece arka plan)."""
        ts = np.empty(n_deneme)
        for i in range(n_deneme):
            ra = rng.uniform(0, 360, size=len(self.ra_bant))
            ts[i] = self.fit(ra)[1]
        return ts


    # ---- Aşama 5: sinyal enjeksiyonu ----
    def sinyal_havuzu_ayarla(self, ev, genislik=5.0):
        """Sahte sinyal olaylarına verilecek açısal belirsizlikler: kaynağın
        Dec'ine yakın gerçek olaylardan çekilir (Aşama 3'teki ders)."""
        yakin = np.abs(ev['dec'].values - self.dec_k) < genislik
        self.sig_havuzu = ev['angErr'].values[yakin]

    def sahte_sinyal_sb(self, n, rng):
        """n tane sahte kaynak olayı üret, S/B oranlarını döndür."""
        sig = rng.choice(self.sig_havuzu, n)
        # 2B Gauss saçılma: kaynaktan uzaklık Rayleigh dağılımına uyar
        psi = sig * np.sqrt(-2 * np.log(rng.uniform(1e-12, 1, n)))
        aci = rng.uniform(0, 2 * np.pi, n)
        dec = np.clip(self.dec_k + psi * np.cos(aci), -89.99, 89.99)
        return sinyal_pdf(psi, sig) / self.B(dec)

    def sinyal_denemeleri(self, ns_ort, n_deneme, rng):
        """Arka plan (RA karıştırılmış) + ortalama ns_ort olan Poisson sayıda sinyal.
        Her deneme için (enjekte edilen n, TS) döndürür."""
        sonuc = np.empty((n_deneme, 2))
        for i in range(n_deneme):
            ra = rng.uniform(0, 360, size=len(self.ra_bant))
            sb_arka = self.sb_orani(ra)
            n = rng.poisson(ns_ort)
            sb = np.concatenate([sb_arka, self.sahte_sinyal_sb(n, rng)]) if n else sb_arka
            sonuc[i] = n, self.fit_sb(sb, self.N + n)[1]
        return sonuc


# --------------------------------------------------------------------------
# Anlamlılık
# --------------------------------------------------------------------------
def p_degeri(ts_gozlenen, ts_arka):
    """Monte Carlo p-değeri ve Wilks (χ²) yaklaşımı.

    Wilks teoremi: H₀ altında TS, yarısı 0, yarısı χ²(1 sd) olacak şekilde dağılır
    (nₛ ≥ 0 sınırı yüzünden). Monte Carlo'nun ulaşamadığı kuyrukta bunu kullanırız.
    """
    n = len(ts_arka)
    p_mc = (np.sum(ts_arka >= ts_gozlenen) + 1) / (n + 1)
    eta = np.mean(ts_arka > 0)                       # sıfırdan büyük TS oranı
    p_wilks = eta * stats.chi2.sf(ts_gozlenen, df=1) if ts_gozlenen > 0 else 1.0
    return p_mc, p_wilks


def sigma(p):
    return stats.norm.isf(p)


# --------------------------------------------------------------------------
# Aşama 5: Nötrino sayısından akıya (dedektör tepki dosyaları)
# --------------------------------------------------------------------------
# Veri setinin "irfs" klasöründeki etkin alan dosyaları:
#   log10(E_nu/GeV)_min  log10(E_nu/GeV)_max  Dec_nu_min[deg]  Dec_nu_max[deg]  A_Eff[cm^2]
# "uptime" klasöründeki dosyalar:  MJD_start[days]  MJD_stop[days]

def _metin_dosyalari(data_dir, desen):
    """Desene uyan gerçek veri dosyaları. Zip'lerden gelen çöp dosyaları
    (__MACOSX klasörü, '._' ile başlayan dosyalar, zip'in kendisi) atlanır."""
    for f in sorted(Path(data_dir).rglob(desen)):
        if (f.name.startswith('._') or '__MACOSX' in f.parts
                or f.suffix.lower() in ('.zip', '.pdf', '.png', '.jpg', '.eps')):
            continue
        if f.is_file():
            yield f


def _tablo_oku(f, sutunlar):
    """Boşlukla ayrılmış sayısal tabloyu oku. UTF-8 değilse latin-1 dene;
    yine de sayısal tablo değilse uyarı verip None döndür."""
    for kodlama in ('utf-8', 'latin-1'):
        try:
            d = pd.read_csv(f, sep=r'\s+', comment='#', header=None, names=sutunlar,
                            encoding=kodlama)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            break
        # '#' ile başlamayan başlık satırları sayıya çevrilemez → atılır
        d = d.apply(pd.to_numeric, errors='coerce').dropna()
        if len(d):
            return d.reset_index(drop=True)
        break
    print(f"UYARI: {f} okunamadı, atlanıyor (sayısal tablo değil).")
    return None


def _sezon_adi(dosya, ek):
    return dosya.stem.replace(ek, '').rstrip('_')


def etkin_alanlari_yukle(data_dir='../data'):
    """{sezon: DataFrame(logE_min, logE_max, dec_min, dec_max, A_cm2)}"""
    alanlar = {}
    for f in _metin_dosyalari(data_dir, 'irfs/*effectiveArea*'):
        d = _tablo_oku(f, ['logE_min', 'logE_max', 'dec_min', 'dec_max', 'A_cm2'])
        if d is not None:
            alanlar[_sezon_adi(f, 'effectiveArea')] = d
    if not alanlar:
        raise FileNotFoundError("irfs/ klasöründe effectiveArea dosyası bulunamadı.")
    return alanlar


def canli_sureler_yukle(data_dir='../data'):
    """{sezon: canlı süre [saniye]} — dedektörün analize uygun çalıştığı toplam süre."""
    sureler = {}
    for f in _metin_dosyalari(data_dir, 'uptime/*'):
        d = _tablo_oku(f, ['bas', 'son'])
        if d is not None:
            sureler[_sezon_adi(f, 'exp')] = float((d['son'] - d['bas']).sum()) * 86400
    if not sureler:
        raise FileNotFoundError("uptime/ klasöründe dosya bulunamadı.")
    return sureler


def _irf_sezonu(sezon, alanlar):
    """IC86_III..VII için ayrı tepki dosyası yoksa IC86_II'ninkini kullan."""
    if sezon in alanlar:
        return sezon
    if sezon.startswith('IC86') and 'IC86_II' in alanlar:
        return 'IC86_II'
    raise KeyError(f"{sezon} için etkin alan yok. Mevcut: {list(alanlar)}")


def aki_basina_olay(dec, gamma, alanlar, sureler, E0=1000.0):
    """
    Φ(E) = Φ0 · (E/E0)^−γ  akısı için, Φ0 = 1 GeV⁻¹ cm⁻² s⁻¹ olsaydı
    beklenen olay sayısı:   N = Σ_sezon T_sezon · ∫ A_eff(E, δ) Φ(E) dE
    Böylece   Φ0 = nₛ / aki_basina_olay(...)
    """
    toplam = 0.0
    for sezon, T in sureler.items():
        a = alanlar[_irf_sezonu(sezon, alanlar)]
        satir = a[(a['dec_min'] <= dec) & (dec < a['dec_max'])]
        E1, E2 = 10.0**satir['logE_min'].values, 10.0**satir['logE_max'].values
        # (E/E0)^−γ'nın kutu içindeki integrali, analitik
        if np.isclose(gamma, 1):
            integral = E0 * np.log(E2 / E1)
        else:
            integral = E0**gamma * (E2**(1 - gamma) - E1**(1 - gamma)) / (1 - gamma)
        toplam += T * np.sum(satir['A_cm2'].values * integral)
    return toplam


def e2_aki(phi0, E0=1000.0):
    """Φ0 [GeV⁻¹ cm⁻² s⁻¹] → E0²·Φ(E0) [TeV cm⁻² s⁻¹] (literatürdeki standart birim)."""
    return phi0 * E0**2 / 1000.0


# --------------------------------------------------------------------------
# Aşama 5: Hassasiyet eğrileri
# --------------------------------------------------------------------------
def esik_gecen_ns(ns_izgara, ts_dizileri, esik, oran):
    """Her ns için 'TS > esik' oranını hesapla; bu oranın `oran`a ulaştığı ns'i
    doğrusal interpolasyonla bul."""
    f = np.array([np.mean(ts > esik) for ts in ts_dizileri])
    # monoton olmayan küçük dalgalanmalara karşı birikimli maksimum
    f_mon = np.maximum.accumulate(f)
    if f_mon[-1] < oran:
        return np.nan, f
    return float(np.interp(oran, f_mon + np.arange(len(f_mon)) * 1e-9, ns_izgara)), f


def ts_5sigma(eta):
    """Wilks ile 5σ'ya karşılık gelen TS eşiği: η·χ²_sf(TS) = p(5σ)."""
    return stats.chi2.isf(stats.norm.sf(5) / eta, df=1)


# --------------------------------------------------------------------------
# Aşama 6: Stacking — birden fazla kaynağın sinyalini tek testte birleştirmek
# --------------------------------------------------------------------------
class StackingLikelihood:
    """
    Tek bir nₛ, ama sinyal yoğunluğu kaynakların ağırlıklı toplamı:
        S_i = Σ_k w_k · S(ψ_ik, σ_i),   Σ w_k = 1
    Ağırlıklar: dedektörün o yöndeki kabulü (Aşama 5'teki akı başına olay sayısı).
    """
    PENCERE = 10.0

    def __init__(self, ev, arka_plan, kaynaklar, agirliklar):
        self.N = len(ev)
        self.kaynaklar = list(kaynaklar.values())
        w = np.asarray(agirliklar, float)
        self.w = w / w.sum()
        dec = ev['dec'].values
        maske = np.zeros(len(ev), bool)
        for _, dk in self.kaynaklar:
            maske |= np.abs(dec - dk) < self.PENCERE
        self.dec, self.ra, self.sig = dec[maske], ev['ra'].values[maske], ev['angErr'].values[maske]
        self.B = arka_plan(self.dec)

    def sb_orani(self, ra=None):
        ra = self.ra if ra is None else ra
        S = np.zeros(len(self.dec))
        for (rk, dk), wk in zip(self.kaynaklar, self.w):
            psi = acisal_uzaklik(ra, self.dec, rk, dk)
            yakin = psi < self.PENCERE
            S[yakin] += wk * sinyal_pdf(psi[yakin], self.sig[yakin])
        yakin = S > 0
        return S[yakin] / self.B[yakin]

    def fit(self, ra=None):
        return ns_fit(self.sb_orani(ra), self.N)

    def arka_plan_denemeleri(self, n_deneme, rng, ilerleme=None):
        ts = np.empty(n_deneme)
        araliklar = range(n_deneme) if ilerleme is None else ilerleme(range(n_deneme))
        for i in araliklar:
            ts[i] = self.fit(rng.uniform(0, 360, len(self.ra)))[1]
        return ts


# --------------------------------------------------------------------------
# Aşama 7: Tüm gökyüzü taraması
# --------------------------------------------------------------------------
def gok_izgarasi(adim=0.7, dec_min=-5.0, dec_max=85.0):
    """Yaklaşık eşit alanlı nokta ızgarası: Dec halkaları `adim` derece aralıklı,
    her halkada RA aralığı adim/cos(Dec). (HEALPix'in basit bir benzeri.)"""
    ra_l, dec_l = [], []
    for d in np.arange(dec_min, dec_max + 1e-9, adim):
        n = max(1, int(round(360 * np.cos(np.radians(d)) / adim)))
        ra_l.append((np.arange(n) + 0.5) * 360 / n)
        dec_l.append(np.full(n, d))
    return np.concatenate(ra_l), np.concatenate(dec_l)


class Tarayici:
    """
    Izgaradaki her noktada nokta kaynak fit'i. Hız için: olaylar Dec halkası
    başına bir kez seçilip RA'ya göre sıralanır, her noktada sadece RA penceresi
    içindekiler (ikili arama ile) hesaplanır.
    """
    PENCERE = 5.0     # derece; σ ≲ 1° olaylarda S bu mesafede sıfıra iner

    def __init__(self, ev, arka_plan):
        self.N = len(ev)
        sira = np.argsort(ev['dec'].values)
        self.dec = ev['dec'].values[sira]
        self.ra = ev['ra'].values[sira]
        self.sig = ev['angErr'].values[sira]
        self.B = arka_plan(self.dec)

    def harita(self, ra_izg, dec_izg, ra_olay=None, ilerleme=None):
        """Her ızgara noktası için (ns, TS). ra_olay verilirse (karıştırılmış gökyüzü)
        olayların RA'ları yerine bu kullanılır."""
        ra_olay = self.ra if ra_olay is None else ra_olay
        ns = np.zeros(len(ra_izg)); ts = np.zeros(len(ra_izg))
        halkalar = np.unique(dec_izg)
        if ilerleme is not None:
            halkalar = ilerleme(halkalar)
        for d in halkalar:
            # Bu halkaya PENCERE derece yakın olaylar (dec sıralı → ikili arama)
            a, b = np.searchsorted(self.dec, [d - self.PENCERE, d + self.PENCERE])
            r, dd, sg, BB = ra_olay[a:b], self.dec[a:b], self.sig[a:b], self.B[a:b]
            o = np.argsort(r)
            r, dd, sg, BB = r[o], dd[o], sg[o], BB[o]
            # RA'da sarmalanma (359° ile 1° komşu) için kopyaları ekle
            r3 = np.concatenate([r - 360, r, r + 360])
            idx3 = np.concatenate([np.arange(len(r))] * 3)
            genislik = min(180.0, self.PENCERE / max(np.cos(np.radians(abs(d) + self.PENCERE)), 1e-3))
            for j in np.where(dec_izg == d)[0]:
                i1, i2 = np.searchsorted(r3, [ra_izg[j] - genislik, ra_izg[j] + genislik])
                k = idx3[i1:i2]
                if genislik >= 180:
                    k = np.unique(k)
                psi = acisal_uzaklik(r[k], dd[k], ra_izg[j], d)
                yakin = psi < self.PENCERE
                sb = sinyal_pdf(psi[yakin], sg[k][yakin]) / BB[k][yakin]
                ns[j], ts[j] = ns_fit(sb, self.N)
        return ns, ts

    def karisik_ra(self, rng):
        return rng.uniform(0, 360, self.N)


def yerel_p(ts, eta=0.5):
    """Wilks ile tek nokta (yerel, 'pre-trial') p-değeri."""
    return np.where(ts > 0, eta * stats.chi2.sf(ts, df=1), 1.0)


def etkin_deneme_sayisi(min_p_karisik):
    """Karıştırılmış haritaların en küçük yerel p'lerinden 'gökyüzünde kaç bağımsız
    nokta var' sayısını (N_eff) en çok olabilirlik ile tahmin et:
        P(min p ≤ x) = 1 − (1 − x)^N_eff"""
    m = np.clip(np.asarray(min_p_karisik), 1e-300, 1 - 1e-12)
    return -len(m) / np.sum(np.log1p(-m))


def galaksi_duzlemi(n=721):
    """Gökada düzleminin (b = 0) ekvatoral koordinatları [derece], J2000."""
    ra_gp, dec_gp, l_ncp = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)
    l = np.radians(np.linspace(0, 360, n))
    b = 0.0
    sin_dec = np.sin(b) * np.sin(dec_gp) + np.cos(b) * np.cos(dec_gp) * np.cos(l_ncp - l)
    dec = np.arcsin(sin_dec)
    y = np.cos(b) * np.sin(l_ncp - l)
    x = np.sin(b) * np.cos(dec_gp) - np.cos(b) * np.sin(dec_gp) * np.cos(l_ncp - l)
    ra = (np.degrees(np.arctan2(y, x) + ra_gp)) % 360
    return ra, np.degrees(dec)


# --------------------------------------------------------------------------
# Aşama 8: Zamana bağlı analiz (patlama/flare araması)
# --------------------------------------------------------------------------
class ZamanAnalizi:
    """
    Kaynaktan gelen nötrinolar zamanda Gauss şeklinde bir patlamada toplanmışsa:
        S_i = S_uzay(ψ_i, σ_i) · G(t_i; T0, σ_T)        B_i = B_uzay(δ_i) · 1/T_canlı
    T0 (patlamanın merkezi) ve σ_T (süresi) bir ızgara üzerinde taranır, her
    noktada nₛ fit edilir. Kısa patlamalar şans eseri daha kolay "bulunduğu"
    için standart bir düzeltme terimi eklenir (Braun ve ark. 2010):
        TS = 2 ln[L(n̂ₛ)/L(0)] − 2 ln[T_canlı / (√(2π) σ_T)]
    """
    PENCERE = 5.0

    def __init__(self, ev, arka_plan, ra_k, dec_k, T_canli_gun,
                 sigma_izgara=(5, 10, 20, 40, 70, 110, 160, 250, 400, 600, 1000, 1500), t0_adim=20.0):
        self.N = len(ev)
        self.ra_k, self.dec_k = ra_k, dec_k
        self.T = float(T_canli_gun)
        bant = np.abs(ev['dec'].values - dec_k) < self.PENCERE
        self.ra = ev['ra'].values[bant]; self.dec = ev['dec'].values[bant]
        self.sig = ev['angErr'].values[bant]; self.t = ev['mjd'].values[bant]
        self.B = arka_plan(self.dec)
        t_hepsi = ev['mjd'].values
        self.t0_izgara = np.arange(t_hepsi.min(), t_hepsi.max() + t0_adim, t0_adim)
        self.sigma_izgara = np.asarray(sigma_izgara, float)

    def uzay_sb(self, ra=None):
        """Kaynağa yakın olayların uzaysal S/B oranları ve zamanları."""
        ra = self.ra if ra is None else ra
        psi = acisal_uzaklik(ra, self.dec, self.ra_k, self.dec_k)
        y = psi < self.PENCERE
        return sinyal_pdf(psi[y], self.sig[y]) / self.B[y], self.t[y]

    def tara(self, ra=None):
        """Tüm (T0, σ_T) ızgarası için TS ve nₛ tabloları."""
        sb_u, t = self.uzay_sb(ra)
        # Patlamaya katkı verebilecek olaylar: uzaysal olarak arka plandan baskın
        guclu = sb_u > 1
        sb_u, t = sb_u[guclu], t[guclu]
        TS = np.full((len(self.sigma_izgara), len(self.t0_izgara)), -np.inf)
        NS = np.zeros_like(TS)
        for a, sT in enumerate(self.sigma_izgara):
            ceza = 2 * np.log(self.T / (np.sqrt(2 * np.pi) * sT))
            for b, t0 in enumerate(self.t0_izgara):
                zaman = np.exp(-0.5 * ((t - t0) / sT) ** 2) / (np.sqrt(2 * np.pi) * sT) * self.T
                k = zaman > 1e-3
                ns, ts = ns_fit(sb_u[k] * zaman[k], self.N)
                NS[a, b], TS[a, b] = ns, ts - ceza
        return NS, TS

    def en_iyi(self, ra=None):
        NS, TS = self.tara(ra)
        a, b = np.unravel_index(np.argmax(TS), TS.shape)
        return dict(TS=TS[a, b], ns=NS[a, b], T0=self.t0_izgara[b],
                    sigma_T=self.sigma_izgara[a]), NS, TS


# --------------------------------------------------------------------------
# Aşama 9: Enerji terimi
# --------------------------------------------------------------------------
# irfs/*smearing* dosyaları: gerçek nötrino enerjisi ve yönü verildiğinde
# dedektörün ölçtüğü enerji, nokta yayılımı ve açısal hatanın dağılımı.
SMEARING_SUTUNLARI = ['logEnu_min', 'logEnu_max', 'dec_min', 'dec_max',
                      'logE_min', 'logE_max', 'psf_min', 'psf_max',
                      'angerr_min', 'angerr_max', 'oran']


def smearing_yukle(data_dir='../data', sezon=None):
    """Bir sezonun smearing tablosu. Sezon verilmezse en güncel olanı."""
    dosyalar = {_sezon_adi(f, 'smearing'): f for f in _metin_dosyalari(data_dir, 'irfs/*smearing*')}
    if not dosyalar:
        raise FileNotFoundError("irfs/ klasöründe smearing dosyası bulunamadı.")
    sezon = sezon or sorted(dosyalar)[-1]
    d = _tablo_oku(dosyalar[sezon], SMEARING_SUTUNLARI)
    if d is None:
        raise ValueError(f"{dosyalar[sezon]} okunamadı; ilk satırlarını kontrol et.")
    return sezon, d


class EnerjiPDF:
    """
    Her olay için enerji oranı  E_i(γ) = P_sinyal(logE | δ, γ) / P_arka(logE | δ)
      - P_arka: verinin kendisinden, sin(δ) dilimlerinde logE histogramı
      - P_sinyal: Φ ∝ E^−γ akısını etkin alan ve smearing tablosuyla
                  "dedektörden geçirip" ölçülen enerji dağılımına çevirerek
    """

    def __init__(self, ev, alan, smearing, gamma_izgara=np.arange(1.5, 4.01, 0.25),
                 logE_kenar=np.arange(1.0, 9.01, 0.25),
                 sindec_kenar=np.concatenate([np.linspace(-1, -0.2, 9)[:-1],    # ufuk çevresinde
                                              np.linspace(-0.2, 0.2, 21)[:-1],  # (−5° seçim sınırı)
                                              np.linspace(0.2, 1, 9)]),         # daha ince dilimler
                 oran_ust=1e4):
        self.gamma = np.asarray(gamma_izgara, float)
        self.logE_kenar, self.sindec_kenar = logE_kenar, sindec_kenar
        nE, nD = len(logE_kenar) - 1, len(sindec_kenar) - 1

        # --- Arka plan: veriden ---
        sayim, _, _ = np.histogram2d(np.sin(np.radians(ev['dec'].values)), ev['logE'].values,
                                     bins=[sindec_kenar, logE_kenar])
        sayim = sayim + 0.5 * (sayim.sum(axis=1, keepdims=True) > 0)   # boş kutulara küçük taban
        self.P_arka = sayim / sayim.sum(axis=1, keepdims=True)

        # --- Sinyal: dedektör simülasyon tablolarından ---
        sm = smearing
        sm_merkez_E = 0.5 * (sm['logE_min'] + sm['logE_max']).values
        sm_merkez_nu = 0.5 * (sm['logEnu_min'] + sm['logEnu_max']).values
        self.P_sinyal = np.zeros((len(self.gamma), nD, nE))
        dec_merkez = np.degrees(np.arcsin(0.5 * (sindec_kenar[1:] + sindec_kenar[:-1])))
        for j, dec in enumerate(dec_merkez):
            satir = sm[(sm['dec_min'] <= dec) & (dec < sm['dec_max'])]
            if satir.empty:
                continue
            idx = satir.index.values
            # bu yöndeki etkin alan, her gerçek-enerji kutusunun merkezinde
            a = alan[(alan['dec_min'] <= dec) & (dec < alan['dec_max'])]
            A = np.interp(sm_merkez_nu[idx], 0.5 * (a['logE_min'] + a['logE_max']).values,
                          a['A_cm2'].values, left=0, right=0)
            E1 = 10.0 ** satir['logEnu_min'].values
            E2 = 10.0 ** satir['logEnu_max'].values
            for g, gam in enumerate(self.gamma):
                aki = (E2 ** (1 - gam) - E1 ** (1 - gam)) / (1 - gam) if gam != 1 else np.log(E2 / E1)
                # Not: aynı gerçek-enerji kutusu smearing'de birçok satıra bölünür;
                # 'oran' sütunu o kutunun olaylarının bu satıra düşen kesridir.
                w = A * aki * satir['oran'].values
                h, _ = np.histogram(sm_merkez_E[idx], bins=logE_kenar, weights=w)
                if h.sum() > 0:
                    self.P_sinyal[g, j] = h / h.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            self.oran_tablo = np.where(self.P_arka[None] > 0, self.P_sinyal / self.P_arka[None], 0)
        self.oran_tablo = np.clip(self.oran_tablo, 0, oran_ust)

    def _kutu(self, dec, logE):
        i = np.clip(np.digitize(np.sin(np.radians(dec)), self.sindec_kenar) - 1, 0, len(self.sindec_kenar) - 2)
        k = np.clip(np.digitize(logE, self.logE_kenar) - 1, 0, len(self.logE_kenar) - 2)
        return i, k

    def oranlar(self, dec, logE):
        """Tüm γ değerleri için enerji oranı: şekil (n_gamma, n_olay)."""
        i, k = self._kutu(dec, logE)
        return self.oran_tablo[:, i, k]

    def oranlar_kaynak(self, dec_kaynak, dec, logE, oran_ust=1e4):
        """Daha doğru sürüm (Aşama 12): sinyalin enerji dağılımı KAYNAĞIN yönünde
        (gerçek yön), arka planınki olayın kendi yönünde değerlendirilir."""
        i, k = self._kutu(dec, logE)
        ik, _ = self._kutu(np.array([dec_kaynak]), np.zeros(1))
        Ps = self.P_sinyal[:, ik[0], :][:, k]                 # (n_gamma, n_olay)
        Pb = self.P_arka[i, k]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(Pb > 0, Ps / Pb, 0)
        return np.clip(r, 0, oran_ust)

    def ornekle(self, g, dec, n, rng):
        """γ = gamma[g] spektrumlu bir kaynaktan n olayın ölçülen enerjisini çek."""
        i, _ = self._kutu(np.full(1, dec), np.zeros(1))
        p = self.P_sinyal[g, i[0]]
        kutu = rng.choice(len(p), size=n, p=p / p.sum())
        return rng.uniform(self.logE_kenar[kutu], self.logE_kenar[kutu + 1])


class EnerjiliLikelihood(NoktaKaynakLikelihood):
    """Aşama 3'teki likelihood + enerji:  S_i/B_i = (uzaysal oran) × E_i(γ).
    nₛ ve γ birlikte fit edilir (γ bir ızgarada taranır)."""

    def __init__(self, ev, arka_plan, enerji_pdf, ra_k, dec_k):
        super().__init__(ev, arka_plan, ra_k, dec_k)
        self.EP = enerji_pdf
        bant = np.abs(ev['dec'].values - dec_k) < self.PENCERE
        self.logE_bant = ev['logE'].values[bant]

    def _uzay(self, ra=None):
        ra = self.ra_bant if ra is None else ra
        psi = self._psi(ra)
        y = psi < self.PENCERE
        return (sinyal_pdf(psi[y], self.sig_bant[y]) / self.B_bant[y],
                self.dec_bant[y], self.logE_bant[y])

    def _gamma_fit(self, sb_uzay, dec, logE, N=None):
        N = self.N if N is None else N
        oran = self.EP.oranlar(dec, logE)              # (n_gamma, n_olay)
        en = (0.0, 0.0, np.nan)
        for g, gam in enumerate(self.EP.gamma):
            ns, ts = ns_fit(sb_uzay * oran[g], N)
            if ts > en[1]:
                en = (ns, ts, gam)
        return en                                       # (ns, TS, γ)

    def fit_enerji(self, ra=None):
        return self._gamma_fit(*self._uzay(ra))

    def arka_plan_denemeleri_enerji(self, n_deneme, rng, ilerleme=None):
        ts = np.empty(n_deneme)
        dongu = range(n_deneme) if ilerleme is None else ilerleme(range(n_deneme))
        for i in dongu:
            ts[i] = self.fit_enerji(rng.uniform(0, 360, len(self.ra_bant)))[1]
        return ts

    def sinyal_denemeleri_enerji(self, ns_ort, gamma_enj, n_deneme, rng, enerji_kullan=True):
        """Sahte sinyal: konum Aşama 5'teki gibi, enerji γ_enj spektrumundan.
        enerji_kullan=False → aynı denemeleri sadece uzaysal likelihood ile değerlendir
        (karşılaştırma için)."""
        g_enj = int(np.argmin(np.abs(self.EP.gamma - gamma_enj)))
        sonuc = np.empty(n_deneme)
        for i in range(n_deneme):
            sb_u, dec_u, logE_u = self._uzay(rng.uniform(0, 360, len(self.ra_bant)))
            n = rng.poisson(ns_ort)
            if n:
                sig = rng.choice(self.sig_havuzu, n)
                psi = sig * np.sqrt(-2 * np.log(rng.uniform(1e-12, 1, n)))
                dec_s = np.clip(self.dec_k + psi * np.cos(rng.uniform(0, 2 * np.pi, n)), -89.99, 89.99)
                sb_u = np.concatenate([sb_u, sinyal_pdf(psi, sig) / self.B(dec_s)])
                dec_u = np.concatenate([dec_u, dec_s])
                logE_u = np.concatenate([logE_u, self.EP.ornekle(g_enj, self.dec_k, n, rng)])
            if enerji_kullan:
                sonuc[i] = self._gamma_fit(sb_u, dec_u, logE_u, self.N + n)[1]
            else:
                sonuc[i] = ns_fit(sb_u, self.N + n)[1]
        return sonuc


# --------------------------------------------------------------------------
# Aşama 10: Fermi-LAT gama ışını kataloğu ile önceden belirlenmiş kaynak listesi
# --------------------------------------------------------------------------
def fermi_katalog_yukle(dosya):
    """4FGL-DR4 FITS dosyasını (gll_psc_v35.fit) okuyup sade bir tabloya çevir."""
    from astropy.io import fits
    with fits.open(dosya) as h:
        d = h[1].data
        tablo = pd.DataFrame({
            'ad': [str(x).strip() for x in d['Source_Name']],
            'iliskili': [str(x).strip() for x in d['ASSOC1']],
            'sinif': [str(x).strip() for x in d['CLASS1']],
            'ra': np.asarray(d['RAJ2000'], float),
            'dec': np.asarray(d['DEJ2000'], float),
            'enerji_akisi': np.asarray(d['Energy_Flux100'], float),   # erg cm⁻² s⁻¹
        })
    return tablo


class KatalogAnalizi:
    """
    Önceden seçilmiş K kaynak için iki test, tek bir karıştırma döngüsünde:
      1) Stacking: tüm kaynaklar tek nₛ ile (Aşama 6 gibi)
      2) Katalog taraması: her kaynak ayrı ayrı; en küçük yerel p, karıştırmalarla düzeltilir
    Hız için olaylar Dec'e göre sıralı tutulur; her kaynakta sadece ±PENCERE bandı işlenir.
    """
    PENCERE = 5.0

    def __init__(self, ev, arka_plan, ra_k, dec_k, agirlik):
        self.N = len(ev)
        sira = np.argsort(ev['dec'].values)
        self.dec = ev['dec'].values[sira]
        self.ra = ev['ra'].values[sira]
        self.sig = ev['angErr'].values[sira]
        self.B = arka_plan(self.dec)
        self.ra_k, self.dec_k = np.asarray(ra_k, float), np.asarray(dec_k, float)
        w = np.asarray(agirlik, float)
        self.w = w / w.sum()
        self.dilim = [tuple(np.searchsorted(self.dec, [d - self.PENCERE, d + self.PENCERE]))
                      for d in self.dec_k]
        # Dec hiç değişmediği için sin/cos'ları bir kere hesapla (hız)
        self._sd, self._cd = np.sin(np.radians(self.dec)), np.cos(np.radians(self.dec))
        self._cos_pencere = np.cos(np.radians(self.PENCERE))

    def analiz(self, ra=None):
        """(stacking_ns, stacking_TS, kaynak_TS_dizisi)"""
        ra = self.ra if ra is None else ra
        S_top = np.zeros(self.N)                    # tüm kaynakların ağırlıklı sinyal yoğunluğu
        ts_k = np.zeros(len(self.ra_k))
        ra_r = np.radians(ra)
        for k, (a, b) in enumerate(self.dilim):
            dk, rk = np.radians(self.dec_k[k]), np.radians(self.ra_k[k])
            cos_psi = self._sd[a:b] * np.sin(dk) + self._cd[a:b] * np.cos(dk) * np.cos(ra_r[a:b] - rk)
            y = np.where(cos_psi > self._cos_pencere)[0]      # önce ucuz eleme, arccos sadece yakınlara
            psi = np.degrees(np.arccos(np.clip(cos_psi[y], -1, 1)))
            S = sinyal_pdf(psi, self.sig[a:b][y])
            ts_k[k] = ns_fit(S / self.B[a:b][y], self.N)[1]
            S_top[a + y] += self.w[k] * S
        idx = np.nonzero(S_top)[0]
        ns, ts = ns_fit(S_top[idx] / self.B[idx], self.N)
        return ns, ts, ts_k

    def karistir(self, rng):
        return rng.uniform(0, 360, self.N)


# --------------------------------------------------------------------------
# Aşama 11: Paralel hesaplama (çok çekirdek)
# --------------------------------------------------------------------------
# Her işçi süreç veriyi bir kere yükler; görevler sadece küçük parametreler taşır.
_ISCI = {}


def _isci_baslat(data_dir):
    ev = veri_yukle(data_dir)
    _ISCI.update(ev=ev, B=ArkaPlan(ev['dec'].values), L={}, T=None, izgara={})


def gorev_nokta_arka_plan(arg):
    """(ra, dec, n_deneme, tohum) → n_deneme tane arka plan TS'si"""
    ra, dec, n, tohum = arg
    anahtar = (ra, dec)
    if anahtar not in _ISCI['L']:
        _ISCI['L'][anahtar] = NoktaKaynakLikelihood(_ISCI['ev'], _ISCI['B'], ra, dec)
    return _ISCI['L'][anahtar].arka_plan_denemeleri(n, np.random.default_rng(tohum))


def gorev_karisik_harita(arg):
    """(adim, tohum) → bir karıştırılmış gökyüzü haritasının en küçük yerel p'si"""
    adim, tohum = arg
    if _ISCI['T'] is None:
        _ISCI['T'] = Tarayici(_ISCI['ev'], _ISCI['B'])
    if adim not in _ISCI['izgara']:
        _ISCI['izgara'][adim] = gok_izgarasi(adim)
    T = _ISCI['T']
    ra_g, dec_g = _ISCI['izgara'][adim]
    _, ts = T.harita(ra_g, dec_g, ra_olay=T.karisik_ra(np.random.default_rng(tohum)))
    return float(yerel_p(ts.max()))


def paralel(gorev, argumanlar, n_islem, data_dir, ilerleme=None, baglam=None):
    """gorev(arg) çağrılarını n_islem sürece dağıt; sonuçları argümanlarla aynı sırada döndür."""
    from concurrent.futures import ProcessPoolExecutor
    argumanlar = list(argumanlar)
    import multiprocessing as mp
    ctx = mp.get_context(baglam) if baglam else None     # Windows'ta varsayılan zaten 'spawn'
    with ProcessPoolExecutor(max_workers=n_islem, initializer=_isci_baslat,
                             initargs=(str(data_dir),), mp_context=ctx) as havuz:
        sonuc = havuz.map(gorev, argumanlar)
        if ilerleme is not None:
            sonuc = ilerleme(sonuc, total=len(argumanlar))
        return list(sonuc)


# --------------------------------------------------------------------------
# Aşama 12: Sezonlu model (her dedektör konfigürasyonu ayrı) + geniş γ aralığı
# --------------------------------------------------------------------------
def ns_fit_genel(x, a, M, c, tol=1e-7, max_iter=80, ust_sinir=1000):
    """
    Sezonlu likelihood için Newton fit'i. Sezon s'teki olay i için:
        log L(nₛ) = Σ_i log(1 + nₛ·aᵢ·xᵢ) + Σ_s M_s·log(1 − nₛ·c_s)
      xᵢ = Sᵢ/Bᵢ − 1,   aᵢ = f_s/N_s (olayın sezonu için),   c_s = f_s/N_s,
      M_s = N_s − (o sezonda kaynağa yakın olay sayısı),   f_s = sinyalin s sezonuna düşen payı
    Tek sezonda (f=1) Aşama 11'deki ns_fit_newton'a indirgenir.
    """
    ax = a * x
    def ll(ns):
        return np.sum(np.log1p(ns * ax)) + np.sum(M * np.log1p(-ns * c))
    def g(ns):
        q = ax / (1 + ns * ax)
        r = c / (1 - ns * c)
        return np.sum(q) - np.sum(M * r), -np.sum(q * q) - np.sum(M * r * r)
    if len(x) == 0 or g(0.0)[0] <= 0:
        return 0.0, 0.0
    # Not: nₛ·c < 1 olmalı (log içi pozitif); yayılmış kaynaklarda nₛ büyük olabilir
    ust = float(min(len(x), ust_sinir, 0.999 / max(c.max(), 1e-300)))
    alt = 0.0
    if g(ust)[0] > 0:
        return ust, max(0.0, 2 * ll(ust))
    ns = 0.0
    for _ in range(max_iter):
        f, df = g(ns)
        if f > 0:
            alt = ns
        else:
            ust = ns
        yeni = ns - f / df
        if not (alt < yeni < ust):
            yeni = 0.5 * (alt + ust)
        if abs(yeni - ns) < tol * max(1.0, ns):
            ns = yeni
            break
        ns = yeni
    return ns, max(0.0, 2 * ll(ns))


def aki_basina_olay_sezon(dec, gamma, alan, T_saniye, E0=1000.0):
    """Tek sezon için Φ0 = 1 GeV⁻¹ cm⁻² s⁻¹ akının beklenen olay sayısı."""
    satir = alan[(alan['dec_min'] <= dec) & (dec < alan['dec_max'])]
    E1, E2 = 10.0 ** satir['logE_min'].values, 10.0 ** satir['logE_max'].values
    if np.isclose(gamma, 1):
        integral = E0 * np.log(E2 / E1)
    else:
        integral = E0 ** gamma * (E2 ** (1 - gamma) - E1 ** (1 - gamma)) / (1 - gamma)
    return T_saniye * np.sum(satir['A_cm2'].values * integral)


class SezonluModel:
    """
    Her sezon (IC40, IC59, IC79, IC86_I, ...) için ayrı:
      - uzaysal arka plan B_s (o sezonun olaylarından),
      - enerji olasılık tabloları (o sezonun olayları + o sezonun smearing/etkin alanı),
      - canlı süre ve etkin alan → sinyalin sezonlara dağılımı f_s(δ, γ).
    """

    def __init__(self, ev, alanlar, sureler, smearingler,
                 gamma_izgara=np.round(np.arange(1.5, 4.51, 0.1), 2),
                 arka_kutu=None, logE_adim=0.25, sezon_sec=None):
        self.gamma = np.asarray(gamma_izgara, float)
        self.sezonlar = sorted(ev['season'].unique()) if sezon_sec is None else list(sezon_sec)
        self.ev, self.B, self.EP, self.alan, self.T, self.N = {}, {}, {}, {}, {}, {}
        for s in self.sezonlar:
            e = ev[ev['season'] == s]
            irf = _irf_sezonu(s, alanlar)
            self.ev[s] = e
            self.N[s] = len(e)
            kutu = arka_kutu or (100 if len(e) > 1e5 else 50)
            self.B[s] = ArkaPlan(e['dec'].values, n_kutu=kutu)
            self.alan[s] = alanlar[irf]
            if s not in sureler:
                raise KeyError(f"{s} sezonunun canlı süresi (uptime) yok. Mevcut: {list(sureler)}")
            self.T[s] = sureler[s]
            sm = smearingler.get(irf, smearingler[sorted(smearingler)[-1]])
            self.EP[s] = EnerjiPDF(e, alanlar[irf], sm, gamma_izgara=self.gamma,
                                   logE_kenar=np.arange(1.0, 9.01, logE_adim))

    def sezon_paylari(self, dec, gamma):
        """f_s: γ spektrumlu bir kaynağın sinyalinin hangi kesri hangi sezona düşer."""
        A = np.array([aki_basina_olay_sezon(dec, gamma, self.alan[s], self.T[s]) for s in self.sezonlar])
        return A / A.sum() if A.sum() > 0 else np.full(len(A), 1 / len(A))

    def aki_basina_olay(self, dec, gamma):
        return sum(aki_basina_olay_sezon(dec, gamma, self.alan[s], self.T[s]) for s in self.sezonlar)


def smearingleri_yukle(data_dir='../data'):
    """{irf sezonu: smearing tablosu}"""
    sonuc = {}
    for f in _metin_dosyalari(data_dir, 'irfs/*smearing*'):
        d = _tablo_oku(f, SMEARING_SUTUNLARI)
        if d is not None:
            sonuc[_sezon_adi(f, 'smearing')] = d
    if not sonuc:
        raise FileNotFoundError('smearing dosyası bulunamadı')
    return sonuc


class SezonluLikelihood:
    """Tek bir kaynak yönü için sezonlu, enerjili nokta kaynak likelihood'u."""
    PENCERE = 10.0

    def __init__(self, model, ra_k, dec_k):
        self.m, self.ra_k, self.dec_k = model, ra_k, dec_k
        self.bant = {}
        for s in model.sezonlar:
            e = model.ev[s]
            k = np.abs(e['dec'].values - dec_k) < self.PENCERE
            d = dict(ra=e['ra'].values[k], dec=e['dec'].values[k], sig=e['angErr'].values[k],
                     logE=e['logE'].values[k], mjd=e['mjd'].values[k])
            d['B'] = model.B[s](d['dec'])
            dr = np.radians(d['dec'])
            d['A'] = np.sin(dr) * np.sin(np.radians(dec_k))
            d['C'] = np.cos(dr) * np.cos(np.radians(dec_k))
            self.bant[s] = d
        # γ'ya göre sezon payları (bir kere hesapla)
        self.f = np.array([model.sezon_paylari(dec_k, g) for g in model.gamma])   # (n_gamma, n_sezon)

    def yakin_olaylar(self, ra_dict=None, ekstra=None):
        """Her sezon için kaynağa yakın olayların (uzaysal S/B, enerji oranları[γ,olay])."""
        sonuc = {}
        for s, d in self.bant.items():
            ra = d['ra'] if ra_dict is None else ra_dict[s]
            cos_psi = d['A'] + d['C'] * np.cos(np.radians(ra - self.ra_k))
            y = cos_psi > np.cos(np.radians(self.PENCERE))
            psi = np.degrees(np.arccos(np.clip(cos_psi[y], -1, 1)))
            sb = sinyal_pdf(psi, d['sig'][y]) / d['B'][y]
            E = self.m.EP[s].oranlar_kaynak(self.dec_k, d['dec'][y], d['logE'][y])
            if ekstra and s in ekstra:                       # enjekte edilmiş sahte olaylar
                sb = np.concatenate([sb, ekstra[s][0]])
                E = np.concatenate([E, ekstra[s][1]], axis=1)
            sonuc[s] = (sb, E)
        return sonuc

    def _fit_gamma(self, yakin, g, N_ek=None):
        x, a, M, c = [], [], [], []
        for j, s in enumerate(self.m.sezonlar):
            sb, E = yakin[s]
            N = self.m.N[s] + (N_ek or {}).get(s, 0)
            f = self.f[g, j]
            x.append(sb * E[g] - 1); a.append(np.full(len(sb), f / N))
            M.append(N - len(sb)); c.append(f / N)
        return ns_fit_genel(np.concatenate(x), np.concatenate(a), np.array(M, float), np.array(c))

    def fit(self, ra_dict=None, ekstra=None, N_ek=None, tum_gamma=False):
        """En iyi (nₛ, γ, TS). tum_gamma=True ise γ ızgarası boyunca (nₛ, TS) dizileri de."""
        yakin = self.yakin_olaylar(ra_dict, ekstra)
        sonuc = np.array([self._fit_gamma(yakin, g, N_ek) for g in range(len(self.m.gamma))])
        k = int(np.argmax(sonuc[:, 1]))
        en = (sonuc[k, 0], self.m.gamma[k], sonuc[k, 1])
        return (en, sonuc) if tum_gamma else en

    def kontur(self, ns_izgara):
        """−2Δln L haritası (γ × nₛ), en iyi noktaya göre."""
        yakin = self.yakin_olaylar()
        ll = np.zeros((len(self.m.gamma), len(ns_izgara)))
        for g in range(len(self.m.gamma)):
            x, a, M, c = [], [], [], []
            for j, s in enumerate(self.m.sezonlar):
                sb, E = yakin[s]; N = self.m.N[s]; f = self.f[g, j]
                x.append(sb * E[g] - 1); a.append(np.full(len(sb), f / N))
                M.append(N - len(sb)); c.append(f / N)
            ax = np.concatenate(a) * np.concatenate(x); M = np.array(M, float); c = np.array(c)
            for k, ns in enumerate(ns_izgara):
                ll[g, k] = np.sum(np.log1p(ns * ax)) + np.sum(M * np.log1p(-ns * c))
        return -2 * (ll - ll.max())

    def karistir(self, rng):
        return {s: rng.uniform(0, 360, len(d['ra'])) for s, d in self.bant.items()}

    def arka_plan_denemeleri(self, n, rng, ilerleme=None):
        ts = np.empty(n)
        dongu = range(n) if ilerleme is None else ilerleme(range(n))
        for i in dongu:
            ts[i] = self.fit(self.karistir(rng))[2]
        return ts

    def sahte_sinyal(self, n, gamma_enj, rng):
        """n sahte olayı sezonlara dağıt; her sezonda konum ve enerji o sezonun tablolarından."""
        g = int(np.argmin(np.abs(self.m.gamma - gamma_enj)))
        dagilim = rng.multinomial(n, self.f[g])
        ekstra, N_ek = {}, {}
        for j, s in enumerate(self.m.sezonlar):
            k = dagilim[j]
            if k == 0:
                continue
            e = self.m.ev[s]
            havuz = e['angErr'].values[np.abs(e['dec'].values - self.dec_k) < 5]
            sig = rng.choice(havuz if len(havuz) else e['angErr'].values, k)
            psi = sig * np.sqrt(-2 * np.log(rng.uniform(1e-12, 1, k)))
            dec = np.clip(self.dec_k + psi * np.cos(rng.uniform(0, 2 * np.pi, k)), -89.99, 89.99)
            logE = self.m.EP[s].ornekle(g, self.dec_k, k, rng)
            ekstra[s] = (sinyal_pdf(psi, sig) / self.m.B[s](dec),
                         self.m.EP[s].oranlar_kaynak(self.dec_k, dec, logE))
            N_ek[s] = k
        return ekstra, N_ek

    def sinyal_denemesi(self, n_ort, gamma_enj, rng):
        ekstra, N_ek = self.sahte_sinyal(rng.poisson(n_ort), gamma_enj, rng)
        return self.fit(self.karistir(rng), ekstra, N_ek)


# --------------------------------------------------------------------------
# Aşama 13: Sezonlu + enerjili tüm gökyüzü taraması
# --------------------------------------------------------------------------
TARAMA_GAMMA = np.round(np.arange(1.5, 4.51, 0.3), 2)      # taramada daha kaba γ ızgarası


class SezonluTarayici:
    """
    Aşama 7'deki tarayıcının sezonlu, enerjili hâli. Hız için:
      - her Dec halkası için enerji oranları ve sezon payları bir kere hesaplanır
        (halkadaki tüm noktalar aynı Dec'te → sinyalin enerji dağılımı aynı),
      - her noktada sadece RA penceresindeki olaylar işlenir.
    """
    PENCERE = 5.0

    def __init__(self, model, gamma_izgara=TARAMA_GAMMA):
        self.m = model
        self.g_idx = np.array([int(np.argmin(np.abs(model.gamma - g))) for g in gamma_izgara])
        self.gamma = model.gamma[self.g_idx]
        self.olay = {}
        for s in model.sezonlar:
            e = model.ev[s]
            o = np.argsort(e['dec'].values)
            self.olay[s] = dict(dec=e['dec'].values[o], ra=e['ra'].values[o],
                                sig=e['angErr'].values[o], logE=e['logE'].values[o])
            self.olay[s]['B'] = model.B[s](self.olay[s]['dec'])
            self.olay[s]['sira'] = o                     # karıştırılmış RA'ları aynı sıraya dizmek için

    def karisik_ra(self, rng):
        return {s: rng.uniform(0, 360, self.m.N[s]) for s in self.m.sezonlar}

    def harita(self, ra_izg, dec_izg, ra_olay=None, ilerleme=None):
        n = len(ra_izg)
        NS, G, TS = np.zeros(n), np.full(n, np.nan), np.zeros(n)
        halkalar = np.unique(dec_izg)
        if ilerleme is not None:
            halkalar = ilerleme(halkalar)
        cos_p = np.cos(np.radians(self.PENCERE))
        for d in halkalar:
            # --- halka başına hazırlık ---
            f = np.array([self.m.sezon_paylari(d, g) for g in self.gamma])     # (n_g, n_sezon)
            hazir = []
            for j, s in enumerate(self.m.sezonlar):
                o = self.olay[s]
                a, b = np.searchsorted(o['dec'], [d - self.PENCERE, d + self.PENCERE])
                ra = (o['ra'] if ra_olay is None else ra_olay[s][o['sira']])[a:b]
                E = self.m.EP[s].oranlar_kaynak(d, o['dec'][a:b], o['logE'][a:b])[self.g_idx]
                srt = np.argsort(ra)
                r = ra[srt]
                hazir.append(dict(r3=np.concatenate([r - 360, r, r + 360]), n=len(r),
                                  dec=o['dec'][a:b][srt], sig=o['sig'][a:b][srt],
                                  B=o['B'][a:b][srt], E=E[:, srt], ra=r, N=self.m.N[s], j=j))
            gen = min(180.0, self.PENCERE / max(np.cos(np.radians(abs(d) + self.PENCERE)), 1e-3))
            sd, cd = np.sin(np.radians(d)), np.cos(np.radians(d))
            # --- halkadaki her nokta ---
            for k in np.where(dec_izg == d)[0]:
                parcalar = []
                for h in hazir:
                    if h['n'] == 0:
                        parcalar.append(None); continue
                    i1, i2 = np.searchsorted(h['r3'], [ra_izg[k] - gen, ra_izg[k] + gen])
                    idx = np.arange(i1, i2) % h['n']
                    if gen >= 180:
                        idx = np.unique(idx)
                    dr = np.radians(h['dec'][idx])
                    cps = np.sin(dr) * sd + np.cos(dr) * cd * np.cos(np.radians(h['ra'][idx] - ra_izg[k]))
                    y = cps > cos_p
                    idx = idx[y]
                    psi = np.degrees(np.arccos(np.clip(cps[y], -1, 1)))
                    parcalar.append((sinyal_pdf(psi, h['sig'][idx]) / h['B'][idx], h['E'][:, idx]))
                en = (0.0, np.nan, 0.0)
                for gi in range(len(self.gamma)):
                    x, a, M, c = [], [], [], []
                    for h, p in zip(hazir, parcalar):
                        fs = f[gi, h['j']]
                        m = 0 if p is None else len(p[0])
                        if m:
                            x.append(p[0] * p[1][gi] - 1); a.append(np.full(m, fs / h['N']))
                        M.append(h['N'] - m); c.append(fs / h['N'])
                    if not x:
                        continue
                    ns, ts = ns_fit_genel(np.concatenate(x), np.concatenate(a), np.array(M, float), np.array(c))
                    if ts > en[2]:
                        en = (ns, self.gamma[gi], ts)
                NS[k], G[k], TS[k] = en
        return NS, G, TS


def ince_tara(T, r0, d0, yaricap=1.0, adim=0.1, ra_olay=None, dec_min=-5.0, dec_max=85.0):
    """(r0, d0) çevresinde ince ızgara. Gerçek ve karıştırılmış haritalara AYNI şekilde
    uygulanır; böylece anlamlılık karşılaştırması adil kalır. (ns, γ, TS, ra, dec) döndürür."""
    decs = np.round(np.arange(max(d0 - yaricap, dec_min), min(d0 + yaricap, dec_max) + 1e-9, adim), 4)
    ra_l, dec_l = [], []
    for d in decs:
        w = yaricap / max(np.cos(np.radians(d)), 1e-3)
        rr = np.arange(r0 - w, r0 + w + 1e-9, adim / max(np.cos(np.radians(d)), 1e-3)) % 360
        ra_l.append(rr); dec_l.append(np.full(len(rr), d))
    ra_i, dec_i = np.concatenate(ra_l), np.concatenate(dec_l)
    ns, g, ts = T.harita(ra_i, dec_i, ra_olay=ra_olay)
    k = int(np.argmax(ts))
    return ns[k], g[k], ts[k], ra_i[k], dec_i[k]


def _isci_baslat_sezonlu(data_dir):
    ev = veri_yukle(data_dir)
    model = SezonluModel(ev, etkin_alanlari_yukle(data_dir), canli_sureler_yukle(data_dir),
                         smearingleri_yukle(data_dir))
    _ISCI.update(T13=SezonluTarayici(model), izgara={})


def gorev_karisik_harita_sezonlu(arg):
    """(adim, tohum) → (ince taramayla en büyük TS, kaba haritanın TS'leri)
    Gerçek gökyüzüne uygulanan işlemin aynısı: kaba ızgara + en sıcak noktada ince tarama."""
    adim, tohum = arg[:2]
    dmin, dmax = (arg[2], arg[3]) if len(arg) > 2 else (-5.0, 85.0)     # Aşama 16: güney için
    anahtar = (adim, dmin, dmax)
    if anahtar not in _ISCI['izgara']:
        _ISCI['izgara'][anahtar] = gok_izgarasi(adim, dmin, dmax)
    T = _ISCI['T13']
    ra_g, dec_g = _ISCI['izgara'][anahtar]
    ra_olay = T.karisik_ra(np.random.default_rng(tohum))
    _, _, ts = T.harita(ra_g, dec_g, ra_olay=ra_olay)
    k = int(np.argmax(ts))
    ts_ince = ince_tara(T, ra_g[k], dec_g[k], ra_olay=ra_olay, dec_min=dmin, dec_max=dmax)[2]
    return float(max(ts_ince, ts[k])), ts


def paralel_sezonlu(argumanlar, n_islem, data_dir, ilerleme=None):
    from concurrent.futures import ProcessPoolExecutor
    argumanlar = list(argumanlar)
    with ProcessPoolExecutor(max_workers=n_islem, initializer=_isci_baslat_sezonlu,
                             initargs=(str(data_dir),)) as havuz:
        sonuc = havuz.map(gorev_karisik_harita_sezonlu, argumanlar)
        if ilerleme is not None:
            sonuc = ilerleme(sonuc, total=len(argumanlar))
        return list(sonuc)


# --------------------------------------------------------------------------
# Aşama 14: Önceden belirlenmiş zaman penceresi (TXS 0506+056, 2014–15)
# --------------------------------------------------------------------------
class ZamanPenceresiLikelihood(SezonluLikelihood):
    """
    Sezonlu, enerjili likelihood + sabit bir zaman profili (kutu ya da Gauss).
      Sinyalin zaman yoğunluğu: g(t)  (takvim gününe göre normalize)
      Arka planınki: sezon boyunca düzgün, 1/aralık_s
    Olay başına zaman oranı: g(t)·aralık_s. Sinyalin sezonlara dağılımı:
      f_s ∝ (etkin alan × canlı süre)_s / aralık_s × ∫_sezon g(t) dt
    Pencere önceden sabit olduğu için "başka zamana bakma" cezası yok.
    """

    def __init__(self, model, ra_k, dec_k, zaman_pdf, zaman_cdf):
        super().__init__(model, ra_k, dec_k)
        self.g = zaman_pdf
        self.aralik = {s: (model.ev[s]['mjd'].min(), model.ev[s]['mjd'].max()) for s in model.sezonlar}
        f = []
        for gam in model.gamma:
            w = []
            for s in model.sezonlar:
                t1, t2 = self.aralik[s]
                pay = zaman_cdf(t2) - zaman_cdf(t1)
                A = aki_basina_olay_sezon(dec_k, gam, model.alan[s], model.T[s])
                w.append(A / (t2 - t1) * pay)
            w = np.array(w)
            f.append(w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w)))
        self.f = np.array(f)

    def yakin_olaylar(self, ra_dict=None, ekstra=None):
        sonuc = {}
        for s, d in self.bant.items():
            ra = d['ra'] if ra_dict is None else ra_dict[s]
            t1, t2 = self.aralik[s]
            zaman = self.g(d['mjd']) * (t2 - t1)
            cos_psi = d['A'] + d['C'] * np.cos(np.radians(ra - self.ra_k))
            y = (cos_psi > np.cos(np.radians(self.PENCERE))) & (zaman > 1e-6)
            psi = np.degrees(np.arccos(np.clip(cos_psi[y], -1, 1)))
            sb = sinyal_pdf(psi, d['sig'][y]) / d['B'][y] * zaman[y]
            E = self.m.EP[s].oranlar_kaynak(self.dec_k, d['dec'][y], d['logE'][y])
            sonuc[s] = (sb, E)
        return sonuc


def kutu_pencere(t1, t2):
    pdf = lambda t: np.where((t >= t1) & (t <= t2), 1.0 / (t2 - t1), 0.0)
    cdf = lambda t: np.clip((t - t1) / (t2 - t1), 0, 1)
    return pdf, cdf


def gauss_pencere(t0, sigma):
    pdf = lambda t: stats.norm.pdf(t, t0, sigma)
    cdf = lambda t: stats.norm.cdf(t, t0, sigma)
    return pdf, cdf


# --------------------------------------------------------------------------
# Aşama 17: Gökada düzlemi (yayılmış kaynak) testi
# --------------------------------------------------------------------------
_R_EKV_GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                       [ 0.4941094279, -0.4448296300,  0.7469822445],
                       [-0.8676661490, -0.1980763734,  0.4559837762]])


def ekvator_to_galaksi(ra, dec):
    """J2000 ekvatoral (RA, Dec) → gökada (l, b), derece."""
    ra, dec = np.radians(ra), np.radians(dec)
    v = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    g = _R_EKV_GAL @ v
    return np.degrees(np.arctan2(g[1], g[0])) % 360, np.degrees(np.arcsin(np.clip(g[2], -1, 1)))


class GalaksiDuzlemiLikelihood:
    """
    Nokta kaynak yerine gökyüzüne yayılmış bir sinyal şablonu: gökada düzlemi boyunca
    galaktik enlemde Gauss şeklinde bir bant (genişlik σ_b), gökada merkezine doğru
    galaktik boylamda hafifçe parlayan (genişlik σ_l). Sinyalin gökyüzü dağılımı:
        S(Ω) ∝ şablon(Ω) × dedektör kabulü(δ, γ)   (gökyüzü üzerinde normalize)
    Sabit γ; nₛ fit edilir. Sezonlu model, enerji terimi dahil.
    """

    def __init__(self, model, gamma=2.7, sigma_b=5.0, sigma_l=None, izgara_adim=1.0):
        self.m = model
        self.g = int(np.argmin(np.abs(model.gamma - gamma)))
        self.gamma = model.gamma[self.g]
        ra_g, dec_g = gok_izgarasi(izgara_adim, -89.5, 89.5)
        alan_el = (np.radians(izgara_adim) ** 2)            # yaklaşık eşit alanlı hücreler
        l, b = ekvator_to_galaksi(ra_g, dec_g)
        sablon = np.exp(-0.5 * (b / sigma_b) ** 2)
        if sigma_l:
            dl = (l + 180) % 360 - 180
            sablon = sablon * np.exp(-0.5 * (dl / sigma_l) ** 2)
        self.S_norm, self.f = {}, {}
        dec_kutu = np.linspace(-90, 90, 181)
        kabul = {s: np.array([aki_basina_olay_sezon(d, self.gamma, model.alan[s], model.T[s])
                              for d in 0.5 * (dec_kutu[1:] + dec_kutu[:-1])]) for s in model.sezonlar}
        toplam = {}
        for s in model.sezonlar:
            k = np.clip(np.digitize(dec_g, dec_kutu) - 1, 0, 179)
            agirlik = sablon * kabul[s][k]
            toplam[s] = np.sum(agirlik) * alan_el
            self.S_norm[s] = (sablon, kabul[s], toplam[s])
        T_top = sum(toplam.values())
        self.f = np.array([toplam[s] / T_top for s in model.sezonlar])
        self.sigma_b, self.sigma_l = sigma_b, sigma_l
        self.dec_kutu = dec_kutu
        # olay başına sabit kısımlar (RA karıştırılınca şablon değeri değişir, kabul ve enerji değişmez)
        self.olay = {}
        for s in model.sezonlar:
            e = model.ev[s]
            k = np.clip(np.digitize(e['dec'].values, dec_kutu) - 1, 0, 179)
            E = model.EP[s].oranlar(e['dec'].values, e['logE'].values)[self.g]
            # Arka plan altında enerji oranının ortalaması 1 olmalı; uç kutulardaki kırpma ve
            # boş kutular bunu biraz bozar. Yayılmış bir sinyalde bu küçük sapma bile tüm
            # gökyüzüne yayıldığı için önemli → her Dec diliminde ortalamayı 1'e normalize et.
            i_dec, _ = model.EP[s]._kutu(e['dec'].values, e['logE'].values)
            ort = np.bincount(i_dec, weights=E, minlength=len(model.EP[s].sindec_kenar) - 1)
            say = np.bincount(i_dec, minlength=len(model.EP[s].sindec_kenar) - 1)
            E = E / np.where(say > 0, ort / np.maximum(say, 1), 1.0)[i_dec]
            self.olay[s] = dict(ra=e['ra'].values, dec=e['dec'].values,
                                kabul=self.S_norm[s][1][k] / self.S_norm[s][2],
                                B=model.B[s](e['dec'].values), E=E)

    def _sablon(self, ra, dec):
        l, b = ekvator_to_galaksi(ra, dec)
        t = np.exp(-0.5 * (b / self.sigma_b) ** 2)
        if self.sigma_l:
            dl = (l + 180) % 360 - 180
            t = t * np.exp(-0.5 * (dl / self.sigma_l) ** 2)
        return t

    def fit(self, ra_dict=None):
        x, a, M, c = [], [], [], []
        for j, s in enumerate(self.m.sezonlar):
            o = self.olay[s]
            ra = o['ra'] if ra_dict is None else ra_dict[s]
            S = self._sablon(ra, o['dec']) * o['kabul']
            y = S > 1e-6 * S.max() if S.max() > 0 else np.zeros(len(S), bool)
            sb = S[y] / o['B'][y] * o['E'][y]
            N = self.m.N[s]; fs = self.f[j]
            x.append(sb - 1); a.append(np.full(y.sum(), fs / N))
            M.append(N - y.sum()); c.append(fs / N)
        return ns_fit_genel(np.concatenate(x), np.concatenate(a), np.array(M, float), np.array(c),
                            ust_sinir=1e6)

    def karistir(self, rng):
        return {s: rng.uniform(0, 360, self.m.N[s]) for s in self.m.sezonlar}
