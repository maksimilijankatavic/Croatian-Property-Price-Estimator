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
  zupanija: string;
  grad_opcina: string;
  naselje: string;
  stambena_povrsina: number;
  broj_soba: number;
  godina_izgradnje?: number;
  kat?: number;
  ima_lift?: boolean;
  ima_balkon?: boolean;
  ima_parking?: boolean;
  novogradnja?: boolean;
}

export interface PredictionResult {
  predicted_price: number;
  price_range: {
    low: number;
    high: number;
  };
  price_per_m2: number;
  feature_contributions: {
    feature: string;
    contribution: number;
    description: string;
  }[];
  comparison_by_city: {
    city: string;
    predicted_price: number;
  }[];
}
