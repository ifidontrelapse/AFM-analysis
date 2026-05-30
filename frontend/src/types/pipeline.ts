export type Modality = "afm" | "sem" | "tem";
export type DetectorName = "log" | "yolo";
export type PipelineMode = "detect" | "baseline" | "segment";

export interface Detection {
  x_px: number;
  y_px: number;
  radius_px: number;
  radius_nm: number;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface ParticleMeasurement {
  x_px: number;
  y_px: number;
  height_nm?: number;
  baseline_nm?: number;
  peak_nm?: number;
  area_nm2?: number;
  radius_nm?: number;
  circularity?: number;
  aspect_ratio?: number;
}

export interface PipelineResult {
  detections: Detection[];
  masks_preview_b64: string;
  measurements: ParticleMeasurement[];
  pixel_size_nm: number | null;
  detector_name: DetectorName;
  mode: PipelineMode;
  modality: Modality;
  particle_count: number;
}

export interface PipelineConfig {
  modality: Modality;
  detector: DetectorName;
  mode: PipelineMode;
  nm_per_pixel?: number;
  log_overlap?: number;
  log_percentile?: number;
  yolo_use_tiling?: boolean;
  yolo_conf?: number;
}
