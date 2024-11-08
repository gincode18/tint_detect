import Home from "@/components/home_page";
import { Videos } from "@/types";

async function getVideos() {
  const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/video`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to fetch videos");
  }
  const data: Videos = await response.json();
  return data.video_ids.map((id) => ({ video_id: id }));
}

export default async function Page() {
  const videos = await getVideos();

  return <Home initialVideos={videos} />;
}
