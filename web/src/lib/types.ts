export interface LocationStats {
  zupanija?: string;
  grad_opcina?: string;
  price_mean: number | null;
  price_median: number | null;
  price_min: number | null;
  price_max: number | null;
  count: number;
  price_std: number | null;
  area_mean: number | null;
  area_median: number | null;
  price_per_m2: number | null;
}

export interface PriceDistribution {
  min: number;
  max: number;
  count: number;
}

export interface AreaDistribution {
  min: number;
  max: number;
  count: number;
}

export interface PricePerM2Distribution {
  min: number;
  max: number;
  count: number;
}

export interface YearDistribution {
  decade: string;
  min: number;
  max: number;
  count: number;
}

export interface RoomsDistribution {
  rooms: number;
  count: number;
}

export interface ScatterPoint {
  stambena_povrsina: number;
  cijena: number;
  zupanija: string;
  grad_opcina: string;
  broj_soba?: number | null;
  godina_izgradnje?: number | null;
}

export interface CityData {
  name: string;
  lat: number;
  lng: number;
  apt_count: number;
  apt_avg_price: number;
  house_count: number;
  house_avg_price: number;
}

export interface TopCity {
  grad_opcina: string;
  price_mean: number | null;
  count: number;
}

export interface PropertyTypeData {
  by_zupanija: LocationStats[];
  by_grad: LocationStats[];
  price_distribution: PriceDistribution[];
  area_distribution: AreaDistribution[];
  price_per_m2_distribution: PricePerM2Distribution[];
  year_distribution?: YearDistribution[];
  rooms_distribution: RoomsDistribution[];
  scatter: ScatterPoint[];
  total_count: number;
  avg_price: number;
  median_price: number;
  min_price: number;
  max_price: number;
  std_price: number;
  avg_area: number;
  median_area: number;
  avg_price_per_m2: number;
  median_price_per_m2: number;
  top_expensive_cities: TopCity[];
  top_affordable_cities: TopCity[];
}

export interface EDAData {
  apartments: PropertyTypeData;
  houses: PropertyTypeData;
  cities: CityData[];
}

// Location hierarchy: zupanija -> grad_opcina -> naselja[]
export type LocationHierarchy = Record<string, Record<string, string[]>>;

export interface FormData {
  apartments: {
    zupanije: string[];
    gradovi: string[];
    naselja: string[];
    location_hierarchy: LocationHierarchy;
    encodings: Record<string, { mapping: Record<string, number>; global_mean: number }>;
    feature_names: string[];
  };
  houses: {
    zupanije: string[];
    gradovi: string[];
    naselja: string[];
    location_hierarchy: LocationHierarchy;
    encodings: Record<string, { mapping: Record<string, number>; global_mean: number }>;
    feature_names: string[];
  };
}

export interface PredictionInput {
  property_type: 'apartments' | 'houses';

  // Location (required)
  zupanija: string;
  grad_opcina?: string;
  naselje?: string;

  // Core numerics
  stambena_povrsina: number;
  broj_soba?: number;
  godina_izgradnje?: number;
  godina_renovacije?: number;
  broj_parkirnih_mjesta?: number;
  broj_etaza?: number;
  energetski_razred?: number; // A+=5, A=4, B=3, C=2, D=1, E=0, F=-1, G=-2
  wc_broj?: number;
  kupaonica_s_wc_broj?: number;

  // Apartments only
  kat?: number;
  ukupni_broj_katova?: number;
  tip_stana?: string; // "u_kući" or "u_stambenoj_zgradi"

  // Houses only
  povrsina_okucnice?: number;
  pogled_more?: number;
  tip_kuce?: string; // "samostojeća", "dvojna_duplex", "u_nizu", "stambeno_poslovna"
  vrsta_gradnje?: string; // "zidana_kuća_beton", "kamena_kuća", "montažna_kuća", "drvena_kuća", "opeka"

  // Binary flags
  podatak_novogradnja?: number;
  podatak_lift?: number;
  balkon_balkon?: number;
  balkon_terasa?: number;
  balkon_lodja_loggia?: number;
  grijanje_klima?: number;
  objekt_bazen?: number;
  objekt_dvoriste_vrt?: number;
  objekt_podrum?: number;
  objekt_rostilj?: number;
  objekt_spremiste?: number;
  objekt_vrtna_kucica?: number;
  objekt_zimski_vrt?: number;

  // Orientation (apartments only)
  orijentacija_istok?: number;
  orijentacija_jug?: number;
  orijentacija_sjever?: number;
  orijentacija_zapad?: number;

  // Funk features
  funk_kamin?: number;
  funk_podno_grijanje?: number;
  funk_alarm?: number;
  funk_sauna?: number;

  // Alt energija
  alt_solarni_paneli?: number;
  alt_toplinske_pumpe?: number;

  // Parking type
  parking_type?: string;

  // Heating system
  grijanje_sustav?: string;

  // Permits
  dozvola_vlasnicki_list?: number;
  dozvola_uporabna_dozvola?: number;
  dozvola_gradevinska_dozvola?: number;
}

export interface PredictionResult {
  predicted_price: number;
  price_per_m2: number;
  price_range: {
    low: number;
    high: number;
  };
  model_version: string;
  features_used: number;
}
