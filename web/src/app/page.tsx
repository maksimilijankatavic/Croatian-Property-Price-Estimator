'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import {
  Home,
  Building2,
  MapPin,
  TrendingUp,
  BarChart3,
  Calculator,
  Info,
  Calendar,
  Ruler,
  DoorOpen,
  Euro
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
} from 'recharts';
import type { EDAData, FormData, CityData } from '@/lib/types';
import { formatPrice, formatNumber } from '@/lib/utils';

// Dynamic import for map (no SSR)
const MapComponent = dynamic(() => import('@/components/Map'), {
  ssr: false,
  loading: () => (
    <div className="h-[400px] bg-[var(--card)] rounded-xl animate-pulse flex items-center justify-center border border-[var(--border)]">
      <span className="text-[var(--muted-foreground)]">Učitavanje karte...</span>
    </div>
  )
});

const CHART_COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  tertiary: '#06b6d4',
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
};

// The 10 main cities (9 cities + Grad Zagreb)
const MAIN_CITIES = [
  'Zagreb', 'Split', 'Rijeka', 'Osijek', 'Zadar', 'Pula',
  'Slavonski Brod', 'Karlovac', 'Sisak', 'Velika Gorica'
];

// Custom tooltip style
const tooltipStyle = {
  contentStyle: {
    background: 'var(--card)',
    border: '1px solid var(--border)',
    borderRadius: '8px'
  },
  labelStyle: { color: 'var(--foreground)' },
  cursor: { fill: 'rgba(59, 130, 246, 0.1)' }
};

export default function HomePage() {
  const [edaData, setEdaData] = useState<EDAData | null>(null);
  const [formData, setFormData] = useState<FormData | null>(null);
  const [loading, setLoading] = useState(true);

  // Form state
  const [propertyType, setPropertyType] = useState<'apartments' | 'houses'>('apartments');
  const [selectedCity, setSelectedCity] = useState<string | null>(null);
  const [formValues, setFormValues] = useState({
    zupanija: '',
    grad_opcina: '',
    naselje: '',
    stambena_povrsina: '',
    broj_soba: '',
    godina_izgradnje: '',
    kat: '',
    ima_lift: false,
    ima_balkon: false,
    ima_parking: false,
    novogradnja: false,
  });

  // Prediction state
  const [prediction, setPrediction] = useState<{
    price: number;
    pricePerM2: number;
    range: { low: number; high: number };
    comparisons: { city: string; price: number }[];
  } | null>(null);
  const [predicting, setPredicting] = useState(false);

  // Load data
  useEffect(() => {
    Promise.all([
      fetch('/data/eda.json').then(r => r.json()),
      fetch('/data/form.json').then(r => r.json()),
    ]).then(([eda, form]) => {
      setEdaData(eda);
      setFormData(form);
      setLoading(false);
    }).catch(err => {
      console.error('Error loading data:', err);
      setLoading(false);
    });
  }, []);

  // Get location hierarchy for current property type
  const locationHierarchy = useMemo(() => {
    if (!formData) return null;
    return formData[propertyType]?.location_hierarchy || null;
  }, [formData, propertyType]);

  // Get available županije
  const availableZupanije = useMemo(() => {
    if (!locationHierarchy) return [];
    return Object.keys(locationHierarchy).sort();
  }, [locationHierarchy]);

  // Get available gradovi based on selected županija (filter out Solin)
  const availableGradovi = useMemo(() => {
    if (!locationHierarchy || !formValues.zupanija) return [];
    const grads = locationHierarchy[formValues.zupanija];
    return grads ? Object.keys(grads).filter(g => g !== 'Solin').sort() : [];
  }, [locationHierarchy, formValues.zupanija]);

  // Get available naselja based on selected grad
  const availableNaselja = useMemo(() => {
    if (!locationHierarchy || !formValues.zupanija || !formValues.grad_opcina) return [];
    const naselja = locationHierarchy[formValues.zupanija]?.[formValues.grad_opcina];
    return naselja ? naselja.sort() : [];
  }, [locationHierarchy, formValues.zupanija, formValues.grad_opcina]);

  // Reset dependent fields when parent changes
  const handleZupanijaChange = (value: string) => {
    setFormValues({
      ...formValues,
      zupanija: value,
      grad_opcina: '',
      naselje: ''
    });
  };

  const handleGradChange = (value: string) => {
    setFormValues({
      ...formValues,
      grad_opcina: value,
      naselje: ''
    });
  };

  // Handle prediction
  const handlePredict = async () => {
    if (!formData || !formValues.stambena_povrsina) return;

    setPredicting(true);

    const encodings = formData[propertyType].encodings;
    const area = parseFloat(formValues.stambena_povrsina);

    const zupanijaEnc = encodings?.zupanija?.mapping?.[formValues.zupanija] || encodings?.zupanija?.global_mean || 12.5;
    const gradEnc = encodings?.grad_opcina?.mapping?.[formValues.grad_opcina] || encodings?.grad_opcina?.global_mean || 12.5;
    const naseljeEnc = encodings?.naselje?.mapping?.[formValues.naselje] || encodings?.naselje?.global_mean || 12.5;

    const avgEnc = (zupanijaEnc + gradEnc + naseljeEnc) / 3;
    const basePrice = Math.exp(avgEnc);
    const areaFactor = area / 80;
    const predictedPrice = basePrice * areaFactor;

    const range = {
      low: Math.round(predictedPrice * 0.85),
      high: Math.round(predictedPrice * 1.15),
    };

    // Compare only with main cities
    const comparisons = Object.entries(encodings?.grad_opcina?.mapping || {})
      .filter(([city]) => city !== formValues.grad_opcina && MAIN_CITIES.includes(city))
      .map(([city, enc]) => ({
        city,
        price: Math.round(Math.exp(enc as number) * areaFactor),
      }))
      .sort((a, b) => b.price - a.price)
      .slice(0, 5);

    setTimeout(() => {
      setPrediction({
        price: Math.round(predictedPrice),
        pricePerM2: Math.round(predictedPrice / area),
        range,
        comparisons,
      });
      setPredicting(false);
    }, 500);
  };

  // Filter cities (exclude Solin)
  const filteredCities = useMemo(() => {
    return edaData?.cities?.filter(city => city.name !== 'Solin') || [];
  }, [edaData]);

  // Get current data based on property type and selected city
  const getCurrentData = () => {
    if (!edaData) return null;
    const data = edaData[propertyType];

    if (selectedCity) {
      // Special handling for Zagreb - filter by zupanija instead of grad_opcina
      const isZagreb = selectedCity === 'Zagreb';
      const cityGrad = isZagreb
        ? data.by_zupanija?.find(d => d.zupanija === 'Grad Zagreb')
        : data.by_grad?.find(d => d.grad_opcina === selectedCity);

      const filteredScatter = isZagreb
        ? data.scatter?.filter(d => d.zupanija === 'Grad Zagreb') || []
        : data.scatter?.filter(d => d.grad_opcina === selectedCity) || [];

      return {
        ...data,
        scatter: filteredScatter,
        selectedCityData: cityGrad,
      };
    }

    return data;
  };

  // Get only main cities for the top lists
  const getMainCitiesData = () => {
    if (!currentData?.by_grad) return { expensive: [], affordable: [] };

    const mainCitiesData = currentData.by_grad.filter(city =>
      MAIN_CITIES.includes(city.grad_opcina || '')
    );

    const sorted = [...mainCitiesData].sort((a, b) => (b.price_mean || 0) - (a.price_mean || 0));

    return {
      expensive: sorted.slice(0, 5),
      affordable: sorted.slice(-5).reverse(),
    };
  };

  const currentData = getCurrentData();
  const mainCitiesData = getMainCitiesData();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
        <div className="gradient-orb-1" />
        <div className="gradient-orb-2" />
        <div className="gradient-orb-3" />
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-[var(--muted-foreground)]">Učitavanje podataka...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Background gradient orbs */}
      <div className="gradient-orb-1" />
      <div className="gradient-orb-2" />
      <div className="gradient-orb-3" />

      {/* Header */}
      <header className="sticky top-0 z-50 bg-[var(--background)]/80 backdrop-blur-lg border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                <Home className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg">Procjenitelj Cijena</h1>
                <p className="text-xs text-[var(--muted-foreground)]">Nekretnine u Hrvatskoj</p>
              </div>
            </div>
            <nav className="hidden md:flex items-center gap-6">
              <a href="#procjena" className="text-sm font-medium hover:text-blue-500 transition-colors">Procjena</a>
              <a href="#karta" className="text-sm font-medium hover:text-blue-500 transition-colors">Karta</a>
              <a href="#analiza" className="text-sm font-medium hover:text-blue-500 transition-colors">Analiza</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section with Form */}
      <section id="procjena" className="gradient-hero py-8 lg:py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-6">
            <h2 className="text-4xl lg:text-5xl font-bold mb-2">
              Koliko vrijedi vaša nekretnina?
            </h2>
            <p className="text-lg text-[var(--muted-foreground)] max-w-2xl mx-auto">
              Unesite karakteristike nekretnine i dobijte procjenu cijene temeljenu na
              stvarnim podacima s hrvatskog tržišta nekretnina.
            </p>
          </div>

          {/* Property Type Toggle */}
          <div className="flex justify-center mb-4">
            <div className="tabs">
              <button
                className={`tab ${propertyType === 'apartments' ? 'active' : ''}`}
                onClick={() => {
                  setPropertyType('apartments');
                  setFormValues({ ...formValues, zupanija: '', grad_opcina: '', naselje: '' });
                }}
              >
                <Building2 className="w-4 h-4 inline-block mr-2" />
                Stanovi
              </button>
              <button
                className={`tab ${propertyType === 'houses' ? 'active' : ''}`}
                onClick={() => {
                  setPropertyType('houses');
                  setFormValues({ ...formValues, zupanija: '', grad_opcina: '', naselje: '' });
                }}
              >
                <Home className="w-4 h-4 inline-block mr-2" />
                Kuće
              </button>
            </div>
          </div>

          {/* Prediction Form */}
          <div className="max-w-4xl mx-auto">
            <div className="card">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Zupanija */}
                <div>
                  <label className="label">Županija</label>
                  <select
                    className="select"
                    value={formValues.zupanija}
                    onChange={(e) => handleZupanijaChange(e.target.value)}
                  >
                    <option value="">Odaberite županiju</option>
                    {availableZupanije.map(z => (
                      <option key={z} value={z}>{z}</option>
                    ))}
                  </select>
                </div>

                {/* Grad */}
                <div>
                  <label className="label">Grad / Općina</label>
                  <select
                    className="select"
                    value={formValues.grad_opcina}
                    onChange={(e) => handleGradChange(e.target.value)}
                    disabled={!formValues.zupanija}
                  >
                    <option value="">Odaberite grad</option>
                    {availableGradovi.map(g => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                  {!formValues.zupanija && (
                    <p className="text-xs text-[var(--muted-foreground)] mt-1">Prvo odaberite županiju</p>
                  )}
                </div>

                {/* Naselje */}
                <div>
                  <label className="label">Naselje / Kvart</label>
                  <select
                    className="select"
                    value={formValues.naselje}
                    onChange={(e) => setFormValues({...formValues, naselje: e.target.value})}
                    disabled={!formValues.grad_opcina || availableNaselja.length === 0}
                  >
                    <option value="">{availableNaselja.length === 0 ? 'Nema dostupnih naselja' : 'Odaberite naselje'}</option>
                    {availableNaselja.map(n => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                  {formValues.grad_opcina && availableNaselja.length === 0 && (
                    <p className="text-xs text-[var(--muted-foreground)] mt-1">Nema dodatnih naselja za ovaj grad</p>
                  )}
                </div>

                {/* Povrsina */}
                <div>
                  <label className="label">Stambena površina (m²)</label>
                  <input
                    type="number"
                    className="input"
                    placeholder="npr. 65"
                    value={formValues.stambena_povrsina}
                    onChange={(e) => setFormValues({...formValues, stambena_povrsina: e.target.value})}
                  />
                </div>

                {/* Broj soba */}
                <div>
                  <label className="label">Broj soba</label>
                  <select
                    className="select"
                    value={formValues.broj_soba}
                    onChange={(e) => setFormValues({...formValues, broj_soba: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    {propertyType === 'apartments' ? (
                      <>
                        <option value="0.5">Garsonijera</option>
                        <option value="1">1-sobni</option>
                        <option value="2">2-sobni</option>
                        <option value="3">3-sobni</option>
                        <option value="4">4-sobni</option>
                        <option value="5">5+ sobni</option>
                      </>
                    ) : (
                      <>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                        <option value="7">7+</option>
                      </>
                    )}
                  </select>
                </div>

                {/* Godina izgradnje */}
                <div>
                  <label className="label">Godina izgradnje</label>
                  <input
                    type="number"
                    className="input"
                    placeholder="npr. 2010"
                    value={formValues.godina_izgradnje}
                    onChange={(e) => setFormValues({...formValues, godina_izgradnje: e.target.value})}
                  />
                </div>

                {/* Kat (only for apartments) */}
                {propertyType === 'apartments' && (
                  <div>
                    <label className="label">Kat</label>
                    <select
                      className="select"
                      value={formValues.kat}
                      onChange={(e) => setFormValues({...formValues, kat: e.target.value})}
                    >
                      <option value="">Odaberite</option>
                      <option value="-1">Suteren</option>
                      <option value="0">Prizemlje</option>
                      <option value="1">1. kat</option>
                      <option value="2">2. kat</option>
                      <option value="3">3. kat</option>
                      <option value="4">4. kat</option>
                      <option value="5">5+ kat</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Checkboxes */}
              <div className="mt-6 flex flex-wrap gap-6">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.novogradnja}
                    onChange={(e) => setFormValues({...formValues, novogradnja: e.target.checked})}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600"
                  />
                  <span className="text-sm">Novogradnja</span>
                </label>
                {propertyType === 'apartments' && (
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formValues.ima_lift}
                      onChange={(e) => setFormValues({...formValues, ima_lift: e.target.checked})}
                      className="w-4 h-4 rounded border-gray-300 text-blue-600"
                    />
                    <span className="text-sm">Lift</span>
                  </label>
                )}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.ima_balkon}
                    onChange={(e) => setFormValues({...formValues, ima_balkon: e.target.checked})}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600"
                  />
                  <span className="text-sm">Balkon / Terasa</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.ima_parking}
                    onChange={(e) => setFormValues({...formValues, ima_parking: e.target.checked})}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600"
                  />
                  <span className="text-sm">Parking</span>
                </label>
              </div>

              {/* Submit Button */}
              <div className="mt-8">
                <button
                  className="btn-primary w-full md:w-auto flex items-center justify-center gap-2"
                  onClick={handlePredict}
                  disabled={predicting || !formValues.stambena_povrsina}
                >
                  {predicting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Izračunavam...
                    </>
                  ) : (
                    <>
                      <Calculator className="w-4 h-4" />
                      Izračunaj cijenu
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Prediction Result */}
            {prediction && (
              <div className="mt-8 animate-fade-in">
                <div className="card bg-gradient-to-br from-blue-950 to-purple-950 border-blue-800">
                  <div className="text-center mb-8">
                    <p className="text-sm text-slate-400 mb-2">Procijenjena cijena</p>
                    <p className="text-5xl font-bold text-blue-400">{formatPrice(prediction.price)}</p>
                    <p className="text-sm text-slate-400 mt-2">
                      Raspon: {formatPrice(prediction.range.low)} - {formatPrice(prediction.range.high)}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="stat-card bg-slate-800/50">
                      <div className="stat-value text-cyan-400">{formatNumber(prediction.pricePerM2)} €</div>
                      <div className="stat-label text-slate-400">Cijena po m²</div>
                    </div>

                    <div className="card bg-slate-800/50 border-slate-700">
                      <h4 className="font-semibold mb-4 flex items-center gap-2 text-slate-200">
                        <MapPin className="w-4 h-4" />
                        Usporedba s drugim gradovima
                      </h4>
                      <div className="space-y-2">
                        {prediction.comparisons.map((comp) => (
                          <div key={comp.city} className="flex justify-between items-center text-sm">
                            <span className="text-slate-300">{comp.city}</span>
                            <span className="font-medium text-slate-200">{formatPrice(comp.price)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <p className="text-xs text-center text-slate-500 mt-6">
                    <Info className="w-3 h-3 inline-block mr-1" />
                    Procjena je bazirana na modelu treniranom na {formatNumber(edaData?.apartments.total_count || 0)} stanova i {formatNumber(edaData?.houses.total_count || 0)} kuća
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Map Section */}
      <section id="karta" className="py-6 lg:py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-5">
            <h2 className="text-3xl font-bold mb-2">Istraži tržište po gradovima</h2>
            <p className="text-[var(--muted-foreground)]">
              Klikni na grad za detaljnu analizu tog područja
            </p>
          </div>

          {/* City legend */}
          <div className="flex flex-wrap gap-3 justify-center mb-6">
            {filteredCities.slice(0, 10).map((city) => (
              <button
                key={city.name}
                onClick={() => {
                  setSelectedCity(city.name === selectedCity ? null : city.name);
                  if (city.name !== selectedCity) {
                    setTimeout(() => {
                      document.getElementById('analiza')?.scrollIntoView({ behavior: 'smooth' });
                    }, 100);
                  }
                }}
                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                  selectedCity === city.name
                    ? 'bg-blue-600 text-white'
                    : 'bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] hover:border-blue-500'
                }`}
              >
                {city.name}
              </button>
            ))}
          </div>

          {selectedCity && (
            <div className="mb-6 flex items-center justify-center gap-4">
              <span className="text-sm text-[var(--muted-foreground)]">Prikazujem podatke za:</span>
              <span className="px-3 py-1 bg-blue-600 text-white rounded-full font-medium text-sm">
                {selectedCity}
              </span>
              <button
                className="text-sm text-blue-400 hover:underline"
                onClick={() => setSelectedCity(null)}
              >
                Prikaži sve
              </button>
            </div>
          )}

          <div className="overflow-hidden">
            <MapComponent
              cities={filteredCities}
              onCitySelect={(city) => setSelectedCity(city === selectedCity ? null : city)}
              selectedCity={selectedCity}
              propertyType={propertyType}
            />
          </div>
        </div>
      </section>

      {/* EDA Section */}
      <section id="analiza" className="py-6 lg:py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-5">
            <h2 className="text-3xl font-bold mb-2">
              Analiza tržišta {selectedCity ? `- ${selectedCity}` : ''}
            </h2>
            <p className="text-[var(--muted-foreground)]">
              {propertyType === 'apartments' ? 'Stanovi' : 'Kuće'} -
              {selectedCity
                ? ` Podaci za odabrani grad`
                : ` ${formatNumber(currentData?.total_count || 0)} nekretnina u bazi`
              }
            </p>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.total_count || 0)}</div>
              <div className="stat-label">Ukupno</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatPrice(currentData?.avg_price || 0)}</div>
              <div className="stat-label">Prosjek</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatPrice(currentData?.median_price || 0)}</div>
              <div className="stat-label">Medijan</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.avg_area || 0)} m²</div>
              <div className="stat-label">Prosj. površina</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.avg_price_per_m2 || 0)} €</div>
              <div className="stat-label">Prosj. €/m²</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.median_price_per_m2 || 0)} €</div>
              <div className="stat-label">Medijan €/m²</div>
            </div>
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-5">
            {/* Price by Zupanija */}
            {!selectedCity && (
              <div className="chart-container">
                <h3 className="chart-title flex items-center gap-2">
                  <Euro className="w-4 h-4 text-blue-500" />
                  Prosječne cijene po županijama
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart
                    data={currentData?.by_zupanija?.slice().sort((a, b) => (b.price_mean || 0) - (a.price_mean || 0)).slice(0, 10)}
                    layout="vertical"
                    margin={{ left: 120, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" tickFormatter={(v) => `${(v/1000).toFixed(0)}k €`} />
                    <YAxis type="category" dataKey="zupanija" tick={{ fontSize: 12 }} width={120} />
                    <Tooltip
                      formatter={(value) => formatPrice(value as number)}
                      {...tooltipStyle}
                    />
                    <Bar dataKey="price_mean" fill={CHART_COLORS.primary} radius={[0, 4, 4, 0]} name="Prosječna cijena" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Price Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-500" />
                Distribucija cijena
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={currentData?.price_distribution?.slice(0, 15)}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.secondary} stopOpacity={0.8}/>
                      <stop offset="95%" stopColor={CHART_COLORS.secondary} stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="min" tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => `${formatPrice(v as number)} - ${formatPrice((v as number) + 50000)}`}
                    {...tooltipStyle}
                  />
                  <Area type="monotone" dataKey="count" stroke={CHART_COLORS.secondary} fill="url(#priceGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Scatter Plot */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <Ruler className="w-4 h-4 text-cyan-500" />
                Površina vs Cijena
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart margin={{ bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="stambena_povrsina" name="Površina" unit=" m²" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                  <YAxis dataKey="cijena" name="Cijena" tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} domain={[0, 'auto']} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--card)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px'
                    }}
                    itemStyle={{ color: CHART_COLORS.tertiary }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '8px', padding: '8px 12px' }}>
                            <p style={{ color: 'var(--foreground)', margin: 0 }}>{data.stambena_povrsina} m²</p>
                            <p style={{ color: CHART_COLORS.tertiary, margin: 0 }}>{formatPrice(data.cijena)}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter data={currentData?.scatter} fill={CHART_COLORS.tertiary} fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Price per m2 Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                Distribucija cijene po m²
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={currentData?.price_per_m2_distribution?.slice(0, 15)}>
                  <defs>
                    <linearGradient id="m2Gradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.8}/>
                      <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="min" tickFormatter={(v) => `${v} €`} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => `${v} - ${(v as number) + 500} €/m²`}
                    {...tooltipStyle}
                  />
                  <Area type="monotone" dataKey="count" stroke={CHART_COLORS.success} fill="url(#m2Gradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Area Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <Ruler className="w-4 h-4 text-orange-500" />
                Distribucija površine
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={currentData?.area_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="min" tickFormatter={(v) => `${v} m²`} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => `${v} - ${(v as number) + 20} m²`}
                    {...tooltipStyle}
                  />
                  <Bar dataKey="count" fill={CHART_COLORS.warning} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Rooms Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <DoorOpen className="w-4 h-4 text-pink-500" />
                Distribucija broja soba
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={currentData?.rooms_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="rooms" tick={{ fontSize: 11 }} tickFormatter={(v) => v === 0.5 ? 'Gars.' : `${v}`} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => v === 0.5 ? 'Garsonijera' : `${v} soba`}
                    {...tooltipStyle}
                  />
                  <Bar dataKey="count" fill="#ec4899" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Year Distribution */}
            {currentData?.year_distribution && currentData.year_distribution.length > 0 && (
              <div className="chart-container">
                <h3 className="chart-title flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-indigo-500" />
                  Distribucija godine izgradnje
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={currentData.year_distribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="decade" tick={{ fontSize: 11 }} />
                    <YAxis />
                    <Tooltip
                      formatter={(value) => [value as number, 'Broj nekretnina']}
                      {...tooltipStyle}
                    />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Price per m2 by Zupanija */}
            {!selectedCity && (
              <div className="chart-container">
                <h3 className="chart-title flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-500" />
                  Cijena po m² po županijama
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart
                    data={currentData?.by_zupanija?.slice().sort((a, b) => (b.price_per_m2 || 0) - (a.price_per_m2 || 0)).slice(0, 10)}
                    layout="vertical"
                    margin={{ left: 120, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" tickFormatter={(v) => `${v.toLocaleString()} €`} />
                    <YAxis type="category" dataKey="zupanija" tick={{ fontSize: 12 }} width={120} />
                    <Tooltip
                      formatter={(value) => `${formatNumber(value as number)} €/m²`}
                      {...tooltipStyle}
                    />
                    <Bar dataKey="price_per_m2" fill={CHART_COLORS.success} radius={[0, 4, 4, 0]} name="€/m²" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Top Cities Tables - only show main cities */}
          {!selectedCity && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Most Expensive */}
              <div className="card">
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-red-500" />
                  Najskuplji gradovi
                </h3>
                <div className="space-y-3">
                  {mainCitiesData.expensive.map((city, i) => (
                    <div key={city.grad_opcina} className="flex items-center justify-between p-3 bg-[var(--secondary)] rounded-lg">
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 bg-red-500/20 text-red-400 rounded-full flex items-center justify-center text-xs font-bold">
                          {i + 1}
                        </span>
                        <span className="font-medium">{city.grad_opcina}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{formatPrice(city.price_mean || 0)}</div>
                        <div className="text-xs text-[var(--muted-foreground)]">{city.count} oglasa</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Most Affordable */}
              <div className="card">
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-500 rotate-180" />
                  Najpovoljniji gradovi
                </h3>
                <div className="space-y-3">
                  {mainCitiesData.affordable.map((city, i) => (
                    <div key={city.grad_opcina} className="flex items-center justify-between p-3 bg-[var(--secondary)] rounded-lg">
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 bg-green-500/20 text-green-400 rounded-full flex items-center justify-center text-xs font-bold">
                          {i + 1}
                        </span>
                        <span className="font-medium">{city.grad_opcina}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{formatPrice(city.price_mean || 0)}</div>
                        <div className="text-xs text-[var(--muted-foreground)]">{city.count} oglasa</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
