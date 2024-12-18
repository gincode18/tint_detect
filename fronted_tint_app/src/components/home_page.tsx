"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload, Video as VideoIcon, AlertCircle } from "lucide-react";
import type { Video } from "@/types";

export default function HomeClient({ initialVideos }: { initialVideos: Video[] }) {
  const [videos, setVideos] = useState<Video[]>(initialVideos);
  const [isUploading, setIsUploading] = useState(false);
  const [pageSize, setPageSize] = useState("10");
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchVideos = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/video`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch videos');
      }
      const data = await response.json();
      setVideos(data.video_ids.map((id: string) => ({ video_id: id })));
    } catch (error) {
      console.error("Error fetching videos:", error);
      setVideos([]); // Set videos to empty array on error
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
      if (!response.ok) {
        throw new Error('Failed to upload video');
      }
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

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200"
    >
      <div className="container mx-auto p-4">
        <motion.h1
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-4xl font-bold mb-6 text-gray-800 text-center py-8"
        >
          Car Tint Analyzer
        </motion.h1>
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mb-8 flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4"
        >
          <input
            type="file"
            accept="video/*"
            onChange={handleUpload}
            className="hidden"
            id="video-upload"
            disabled={isUploading}
            ref={fileInputRef}
          />
          <Button
            onClick={handleUploadClick}
            disabled={isUploading}
            className="bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
          >
            <Upload className="mr-2 h-4 w-4" />
            {isUploading ? (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                Uploading...
              </motion.span>
            ) : (
              "Upload Video"
            )}
          </Button>
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
        </motion.div>
        <motion.h2
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="text-2xl font-semibold mb-4 text-gray-700 text-center"
        >
          Recently Uploaded Videos
        </motion.h2>
        <AnimatePresence>
          {videos.length > 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {videos.map((video, index) => (
                <motion.div
                  key={video.video_id}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -50 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <Card
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
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              transition={{ duration: 0.5 }}
              className="text-center p-8 bg-white rounded-lg shadow"
            >
              <AlertCircle className="mx-auto h-12 w-12 text-yellow-500 mb-4" />
              <h3 className="text-xl font-semibold text-gray-800 mb-2">No Videos Available</h3>
              <p className="text-gray-600">
                There are currently no videos to display. This could be due to a connection issue with the backend server or because no videos have been uploaded yet.
              </p>
              <Button
                onClick={fetchVideos}
                className="mt-4 bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
              >
                Retry Loading Videos
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}