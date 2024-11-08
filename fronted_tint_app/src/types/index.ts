export interface Video {
    video_id: string;
  }
  
  export interface Videos {
    video_ids: string[];
  }
  
  export interface CarImage {
    image_id: string;
    url: string;
  }
  
  export interface Pagination {
    current_page: number;
    page_size: number;
    total_images: number;
    total_pages: number;
    has_next: boolean;
    has_previous: boolean;
  }
  
  export interface VideoDetails {
    video_id: string;
    car_images: CarImage[];
    pagination: Pagination;
  }