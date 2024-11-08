"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload, Video as VideoIcon } from "lucide-react";
import type { Videos, Video } from "@/types";

export default function Home() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [pageSize, setPageSize] = useState("10");
  const router = useRouter();

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/video`
      );
      const data: Videos = await response.json();
      setVideos(data.video_ids.map((id) => ({ video_id: id })));
    } catch (error) {
      console.error("Error fetching videos:", error);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/upload_video/`,
        {
          method: "POST",
          body: formData,
        }
      );
      const data = await response.json();
      console.log("Upload successful:", data);
      fetchVideos();
    } catch (error) {
      console.error("Error uploading video:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleVideoClick = (videoId: string) => {
    router.push(`/video/${videoId}?page=1&page_size=${pageSize}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200">
      <div className="container mx-auto p-4">
        <h1 className="text-4xl font-bold mb-6 text-gray-800 text-center py-8">
          Car Tint Analyzer
        </h1>
        <div className="mb-8 flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
          <input
            type="file"
            accept="video/*"
            onChange={handleUpload}
            className="hidden"
            id="video-upload"
            disabled={isUploading}
          />
          <label htmlFor="video-upload">
            <Button
              // as="span"
              disabled={isUploading}
              className="bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
            >
              <Upload className="mr-2 h-4 w-4" />
              {isUploading ? "Uploading..." : "Upload Video"}
            </Button>
          </label>
          <Select value={pageSize} onValueChange={setPageSize}>
            <SelectTrigger className="w-[180px] bg-white text-gray-800 border-gray-300">
              <SelectValue placeholder="Images per page" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="10">10 images per page</SelectItem>
              <SelectItem value="20">20 images per page</SelectItem>
              <SelectItem value="50">50 images per page</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <h2 className="text-2xl font-semibold mb-4 text-gray-700 text-center">
          Recently Uploaded Videos
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {videos.map((video) => (
            <Card
              key={video.video_id}
              className="cursor-pointer hover:shadow-lg transition-all duration-200 transform hover:scale-105 bg-white"
              onClick={() => handleVideoClick(video.video_id)}
            >
              <CardContent className="p-6">
                <img
                  src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/videos/${video.video_id}/thumbnail.png`}
                  alt={`Car ${video.video_id}`}
                  className="w-full h-48 object-cover mb-2 rounded-lg"
                />
                <div className="flex items-center justify-center p-6">
                  <VideoIcon className="h-12 w-12 text-blue-500" />
                  <span className="ml-2 font-medium text-gray-700">
                    Video {video.video_id.slice(-6)}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}