"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChevronLeft, ChevronRight } from "lucide-react";
import type { VideoDetails, CarImage } from "@/types";

export default function VideoPage() {
  const { id } = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [videoDetails, setVideoDetails] = useState<VideoDetails | null>(null);
  const [tintLevels, setTintLevels] = useState<Record<string, string>>({});

  useEffect(() => {
    const page = searchParams.get("page") || "1";
    const pageSize = searchParams.get("page_size") || "10";
    fetchImages(page, pageSize);
  }, [id, searchParams]);

  const fetchImages = async (page: string, pageSize: string) => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/video/${id}?page=${page}&page_size=${pageSize}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch images");
      }
      const data: VideoDetails = await response.json();
      setVideoDetails(data);
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  const fetchTintLevel = async (imageId: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/tint/${imageId}`);
      const data = await response.json();
      setTintLevels((prev) => ({ ...prev, [imageId]: data.tint_level }));
    } catch (error) {
      console.error("Error fetching tint level:", error);
    }
  };

  const changePage = (newPage: number) => {
    router.push(
      `/video/${id}?page=${newPage}&page_size=${videoDetails?.pagination.page_size}`
    );
  };

  if (!videoDetails) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200">
      <div className="container mx-auto p-4">
        <h1 className="text-4xl font-bold mb-6 text-gray-800 text-center py-8">
          Images from Video {id}
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {videoDetails.car_images.map((image: CarImage) => (
            <Card
              key={image.image_id}
              className="overflow-hidden bg-white shadow-md"
            >
              <CardContent className="p-4">
                <img
                  src={image.url}
                  alt={`Car ${image.image_id}`}
                  className="w-full h-48 object-cover mb-2 rounded-lg"
                />
                <Button
                  onClick={() => fetchTintLevel(image.image_id)}
                  className="w-full bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
                >
                  {tintLevels[image.image_id]
                    ? `Tint Level: ${tintLevels[image.image_id]}`
                    : "Check Tint Level"}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="flex flex-col items-center space-y-4">
          <div className="flex justify-center items-center space-x-2">
            <Button
              onClick={() =>
                changePage(videoDetails.pagination.current_page - 1)
              }
              disabled={!videoDetails.pagination.has_previous}
              className="bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <span className="text-lg font-medium text-gray-700">
              Page {videoDetails.pagination.current_page} of{" "}
              {videoDetails.pagination.total_pages}
            </span>
            <Button
              onClick={() =>
                changePage(videoDetails.pagination.current_page + 1)
              }
              disabled={!videoDetails.pagination.has_next}
              className="bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
          <div className="text-sm text-gray-600">
            Showing{" "}
            {(videoDetails.pagination.current_page - 1) *
              videoDetails.pagination.page_size +
              1}{" "}
            -{" "}
            {Math.min(
              videoDetails.pagination.current_page *
                videoDetails.pagination.page_size,
              videoDetails.pagination.total_images
            )}{" "}
            of {videoDetails.pagination.total_images} images
          </div>
        </div>
      </div>
    </div>
  );
}
