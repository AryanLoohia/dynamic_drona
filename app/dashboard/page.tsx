// /app/dashboard/page.tsx
"use client"
import ProtectedRoute from "../components/ProtectedRoutes";
import Link from "next/link"
import Navbar from "../components/Navbar";


import React, { useEffect } from "react";

import {
  useMotionTemplate,
  useMotionValue,
  motion,
  animate,
} from "framer-motion";

const COLORS_TOP = ["#FF4C4C", "#D72638", "#E63946", "#FF6B6B"];

export default function Dashboard() {

    const color = useMotionValue(COLORS_TOP[0]);

  useEffect(() => {
    animate(color, COLORS_TOP, {
      ease: "easeInOut",
      duration: 10,
      repeat: Infinity,
      repeatType: "mirror",
    });
  }, []);

  const backgroundImage = useMotionTemplate`radial-gradient(125% 125% at 50% 0%, #020617 50%, ${color})`;


  return (

    <ProtectedRoute>
        <motion.section
      style={{
        backgroundImage,
      }}
      className="relative grid min-h-screen place-content-center overflow-hidden bg-gray-950 px-4 py-24 text-gray-200"
    >
      <div className="min-h-screen flex flex-col items-center justify-center">
      <Navbar />
      
<h1 className="mb-4 text-4xl text-center font-extrabold leading-none tracking-tight text-gray-900 md:text-5xl lg:text-6xl dark:text-white">Employ our <mark className="px-2 text-white bg-red-600 rounded-sm dark:bg-red-500">Computer Vision</mark> based methods to enhance construction site safety</h1>

      <div className="mt-8 space-y-4 gap-4">
        <Link href="/statics">
          <button className="px-6 mx-5 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600">
            Static Processing
          </button>
        </Link>
        <Link href="/streaming">
          <button className="px-6 mx-5 py-3 bg-red-500 text-red rounded-lg hover:bg-white-600">
            Live Streaming
          </button>
        </Link>
      </div>
    </div>
    </motion.section>
    </ProtectedRoute>
  );
}
