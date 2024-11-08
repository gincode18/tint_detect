import VideoPage from "@/components/video_page";
import { VideoDetails } from "@/types";

async function getVideoDetails(id: string, page: string, pageSize: string) {
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}/video/${id}?page=${page}&page_size=${pageSize}`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error("Failed to fetch video details");
  }
  return response.json();
}

export default async function Page({
  params,
  searchParams,
}: {
  params: { id: string };
  searchParams: { page?: string; page_size?: string };
}) {
  const { id } = params;
  const page = searchParams.page || "1";
  const pageSize = searchParams.page_size || "10";

  const videoDetails: VideoDetails = await getVideoDetails(id, page, pageSize);

  return <VideoPage initialVideoDetails={videoDetails} />;
}
