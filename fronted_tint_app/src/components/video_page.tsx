"use client";

import { useState, useEffect, useRef } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChevronLeft, ChevronRight, Home, X } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { VideoDetails, CarImage } from "@/types";

export default function VideoPage({
  initialVideoDetails,
}: {
  initialVideoDetails: VideoDetails;
}) {
  const { id } = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [videoDetails, setVideoDetails] =
    useState<VideoDetails>(initialVideoDetails);
  const [selectedImage, setSelectedImage] = useState<CarImage | null>(null);
  const [tintLevel, setTintLevel] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchImages = async (page: number) => {
    setIsLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/video/${id}?page=${page}&page_size=${videoDetails.pagination.page_size}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch images");
      }
      const data: VideoDetails = await response.json();
      setVideoDetails(data);
      setCurrentPage(page);
      updateUrlWithoutReload(page);
    } catch (error) {
      console.error("Error fetching images:", error);
      setError("Failed to fetch images. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTintLevel = async (imageId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/tint/${imageId}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch tint level");
      }
      const data = await response.json();
      setTintLevel(data.tint_level);
    } catch (error) {
      console.error("Error fetching tint level:", error);
      setError("Failed to fetch tint level. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const changePage = (newPage: number) => {
    if (newPage !== currentPage) {
      fetchImages(newPage);
    }
  };

  const openModal = (image: CarImage) => {
    setSelectedImage(image);
    setTintLevel(null);
    setError(null);
  };

  const closeModal = () => {
    setSelectedImage(null);
    setTintLevel(null);
    setError(null);
  };

  const updateUrlWithoutReload = (page: number) => {
    const newUrl = `${window.location.pathname}?page=${page}&page_size=${videoDetails.pagination.page_size}`;
    window.history.pushState({ page }, "", newUrl);
  };

  useEffect(() => {
    const page = Number(searchParams.get("page")) || 1;
    setCurrentPage(page);
    fetchImages(page);
  }, [searchParams]);

  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      if (event.state && event.state.page) {
        fetchImages(event.state.page);
      }
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  if (!videoDetails) {
    return <div>Loading...</div>;
  }

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200"
      ref={containerRef}
    >
      <div className="container mx-auto p-4">
        <div className="flex justify-between items-center mb-6">
          <Link href="/">
            <Button
              variant="outline"
              className="flex items-center space-x-2 bg-white hover:bg-gray-100 transition-colors duration-200"
            >
              <Home className="h-4 w-4" />
              <span>Home</span>
            </Button>
          </Link>
          <h1 className="text-4xl font-bold text-gray-800 text-center">
            Images from Video {id}
          </h1>
          <div className="w-[100px]"></div> {/* Spacer for alignment */}
        </div>
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPage}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8"
          >
            {videoDetails.car_images.map((image: CarImage, index) => (
              <motion.div
                key={image.image_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <Card className="overflow-hidden bg-white shadow-md hover:shadow-lg transition-shadow duration-200">
                  <CardContent className="p-4">
                    <img
                      src={image.url}
                      alt={`Car ${image.image_id}`}
                      className="w-full h-48 object-cover mb-2 rounded-lg"
                    />
                    <Button
                      onClick={() => openModal(image)}
                      className="w-full bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
                    >
                      Check Tint Level
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex flex-col items-center space-y-4"
        >
          <div className="flex justify-center items-center space-x-2">
            <Button
              onClick={() => changePage(currentPage - 1)}
              disabled={!videoDetails.pagination.has_previous || isLoading}
              className="bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              Previous
            </Button>
            <span className="text-lg font-medium text-gray-700">
              Page {currentPage} of {videoDetails.pagination.total_pages}
            </span>
            <Button
              onClick={() => changePage(currentPage + 1)}
              disabled={!videoDetails.pagination.has_next || isLoading}
              className="bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
            >
              Next
              <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
          <div className="text-sm text-gray-600">
            Showing {(currentPage - 1) * videoDetails.pagination.page_size + 1}{" "}
            -{" "}
            {Math.min(
              currentPage * videoDetails.pagination.page_size,
              videoDetails.pagination.total_images
            )}{" "}
            of {videoDetails.pagination.total_images} images
          </div>
        </motion.div>
      </div>

      <AnimatePresence>
        {selectedImage && (
          <Dialog open={!!selectedImage} onOpenChange={closeModal}>
            <DialogContent className="sm:max-w-[425px]">
              <DialogHeader>
                <DialogTitle>Tint Level Check</DialogTitle>
                <DialogDescription>
                  Checking tint level for image {selectedImage.image_id}
                </DialogDescription>
              </DialogHeader>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="mt-4"
              >
                <img
                  src={selectedImage.url}
                  alt={`Car ${selectedImage.image_id}`}
                  className="w-full h-48 object-cover mb-4 rounded-lg"
                />
                {isLoading ? (
                  <div className="text-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                      className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"
                    ></motion.div>
                    <p className="mt-2">Loading tint level...</p>
                  </div>
                ) : error ? (
                  <Alert variant="destructive">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                ) : tintLevel ? (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.3 }}
                    className="text-center text-xl font-bold"
                  >
                    Tint Level: {tintLevel}
                  </motion.div>
                ) : (
                  <Button
                    onClick={() =>
                      selectedImage && fetchTintLevel(selectedImage.image_id)
                    }
                    className="w-full bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
                  >
                    Check Tint Level
                  </Button>
                )}
              </motion.div>
            </DialogContent>
          </Dialog>
        )}
      </AnimatePresence>
    </div>
  );
}
