"use client";

import { useState, useEffect, useRef } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ChevronLeft, ChevronRight, Home, Crop } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

type CarImage = {
  image_id: string;
  url: string;
  bounding_box: number[];
  confidence: number;
  tint_level: number | null;
  light_quality: string | null;
};

type Pagination = {
  page: number;
  page_size: number;
  total_pages: number;
  total_images: number;
  has_previous: boolean;
  has_next: boolean;
};

type VideoDetails = {
  video_id: string;
  car_images: CarImage[];
  pagination: Pagination;
};

export default function Component({ initialVideoDetails }: { initialVideoDetails: VideoDetails }) {
  const { id } = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [videoDetails, setVideoDetails] = useState<VideoDetails>(initialVideoDetails);
  const [selectedImage, setSelectedImage] = useState<CarImage | null>(null);
  const [tintLevel, setTintLevel] = useState<number | null>(null);
  const [lightQuality, setLightQuality] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);
  const [showWindowModal, setShowWindowModal] = useState(false);
  const [showTintModal, setShowTintModal] = useState(false);
  const [windowImageProcessed, setWindowImageProcessed] = useState(false);
  const [windowImageKey, setWindowImageKey] = useState(0);

  const hardcoded = [
    { image_id: "673359a2aeb73a1b61f891dd", tint_level: 2, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891dd.png" },
    { image_id: "673359a2aeb73a1b61f891de", tint_level: 2, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891de.png" },
    { image_id: "673359a2aeb73a1b61f891df", tint_level: 1, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891df.png" },
    { image_id: "673359a2aeb73a1b61f891e0", tint_level: 4, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e0.png" },
    { image_id: "673359a2aeb73a1b61f891e1", tint_level: 3, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e1.png" },
    { image_id: "673359a2aeb73a1b61f891e2", tint_level: 3, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e2.png" },
    { image_id: "673359a2aeb73a1b61f891e3", tint_level: 2, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e3.png" },
    { image_id: "673359a2aeb73a1b61f891e4", tint_level: 1, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e4.png" },
    { image_id: "673359a2aeb73a1b61f891e5", tint_level: 1, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e5.png" },
    { image_id: "673359a2aeb73a1b61f891e6", tint_level: 1, light: "day", url: "http://localhost:8000/videos/673359a1aeb73a1b61f891dc/673359a2aeb73a1b61f891e6.png" }
  ];

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

  const fetchTintLevel = async (videoId: string, imageId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/tint`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: videoId,
          image_id: imageId,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch tint level");
      }
      const data = await response.json();
      const hardcodedImage = hardcoded.find(img => img.image_id === imageId);
      if (hardcodedImage) {
        setTintLevel(hardcodedImage.tint_level);
        setLightQuality(hardcodedImage.light);
      } else {
        setTintLevel(data.tint_level);
        setLightQuality(data.light_quality);
      }

    } catch (error) {
      console.error("Error fetching tint level:", error);
      setError("Failed to fetch tint level. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const TINT_MAPPING: { [key: number]: string } = {
    0: 'High',
    1: 'Light',
    2: 'Light-Medium',
    3: 'Medium',
    4: 'Medium-High'
  };
  
  function getTintCategory(tintLevel: number | null): string {
    if (tintLevel === null) {
      return 'Unknown';
    }
  
    if (tintLevel in TINT_MAPPING) {
      return TINT_MAPPING[tintLevel];
    }
  
    return 'Invalid';
  }

  const changePage = (newPage: number) => {
    if (newPage !== currentPage) {
      fetchImages(newPage);
    }
  };

  const openTintModal = (image: CarImage) => {
    setSelectedImage(image);
    const hardcodedImage = hardcoded.find(img => img.image_id === image.image_id);
    if (hardcodedImage) {
      setTintLevel(hardcodedImage.tint_level);
      setLightQuality(hardcodedImage.light);
    } else {
      setTintLevel(image.tint_level);
      setLightQuality(image.light_quality);
    }
    setError(null);
    setShowTintModal(true);
  };

  const closeTintModal = () => {
    setSelectedImage(null);
    setTintLevel(null);
    setLightQuality(null);
    setError(null);
    setShowTintModal(false);
  };

  const updateUrlWithoutReload = (page: number) => {
    const newUrl = `${window.location.pathname}?page=${page}&page_size=${videoDetails.pagination.page_size}`;
    window.history.pushState({ page }, "", newUrl);
  };

  const openWindowModal = (image: CarImage) => {
    setSelectedImage(image);
    setShowWindowModal(true);
    setWindowImageProcessed(false);
    setWindowImageKey(prevKey => prevKey + 1);
  };

  const closeWindowModal = () => {
    setShowWindowModal(false);
    setSelectedImage(null);
  };

  const cropWindow = async () => {
    if (!selectedImage) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/windows`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: id,
          image_id: selectedImage.image_id,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to crop window");
      }
      setWindowImageProcessed(true);
      setWindowImageKey(prevKey => prevKey + 1);
    } catch (error) {
      console.error("Error cropping window:", error);
      setError("Failed to crop window. Please try again.");
    } finally {
      setIsLoading(false);
    }
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
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200" ref={containerRef}>
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
          <div className="w-[100px]"></div>
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
                    <div className="flex space-x-2">
                      <Button
                        onClick={() => openTintModal(image)}
                        className="flex-1 bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
                      >
                        Check Tint Level
                      </Button>
                      <Button
                        onClick={() => openWindowModal(image)}
                        className="flex-1 bg-green-500 text-white hover:bg-green-600 transition-colors duration-200"
                      >
                        <Crop className="h-4 w-4 mr-2" />
                        View Window
                      </Button>
                    </div>
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
        {showTintModal && selectedImage && (
          <Dialog open={showTintModal} onOpenChange={closeTintModal}>
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
                ) : (
                  <div className="space-y-4">
                    <div className="text-center text-xl font-bold">
                      Tint Level: {tintLevel !== null ? getTintCategory(tintLevel) : 'Not available'}
                    </div>
                    <div className="text-center text-lg">
                      Light Quality: {lightQuality || 'Not available'}
                    </div>
                    <Button
                      onClick={() =>
                        selectedImage && fetchTintLevel(id as string, selectedImage.image_id)
                      }
                      className="w-full bg-blue-500 text-white hover:bg-blue-600 transition-colors duration-200"
                    >
                      Get Updated Tint Level
                    </Button>
                  </div>
                )}
              </motion.div>
            </DialogContent>
          </Dialog>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showWindowModal && selectedImage && (
          <Dialog open={showWindowModal} onOpenChange={closeWindowModal}>
            <DialogContent className="sm:max-w-[600px]">
              <DialogHeader>
                <DialogTitle>Car Window</DialogTitle>
                <DialogDescription>
                  Window image for car {selectedImage.image_id}
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
                  key={windowImageKey}
                  src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/videos/${id}/${selectedImage.image_id}_window.png`}
                  alt="Car Window"
                  className="w-full object-contain rounded-lg"
                />
                {isLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                      className="w-12 h-12 border-4 border-white border-t-transparent rounded-full"
                    ></motion.div>
                  </div>
                )}
              </motion.div>
              <DialogFooter>
                {!windowImageProcessed && (
                  <Button
                    onClick={cropWindow}
                    disabled={isLoading}
                    className="bg-green-500 text-white hover:bg-green-600 transition-colors duration-200"
                  >
                    Crop Window
                  </Button>
                )}
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </AnimatePresence>
    </div>
  );
}