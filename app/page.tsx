"use client"
// import Image from "next/image";
import Link from "next/link";

// export default function Home() {
//   return (
//     <div className="min-h-[100vh] w-[100vw]">
//     <section className="h-screen items-center justify-center relative pt-[30vh] bg-white lg:grid lg:pt-0 lg:h-screen lg:place-content-center dark:bg-zinc-900">
//   <div className="mx-auto w-screen max-w-screen-xl px-4 py-16 sm:px-6 sm:py-24 lg:px-8 lg:py-32">
//     <div className="mx-auto max-w-prose text-center">
//       <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl dark:text-white">
//         DYNAMIC HACKATHON<br></br>
//         <strong className="text-red-600"> Team </strong>
//         DRONA
//       </h1>

//       <p className="mt-4 text-base text-pretty text-gray-700 sm:text-lg/relaxed dark:text-gray-200">
//        Dive in and discover our novel solutions assisted by computer vision driving safety
//        and smooth operations at construction sites
//       </p>

//       <div className="mt-4 flex justify-center gap-4 sm:mt-6">
//         <Link
//           className="inline-block rounded border border-red-600 bg-red-600 px-5 py-3 font-medium text-white shadow-sm transition-colors hover:bg-red-700"
//           href="/auth"
//         >
//           <button>Get Started</button>
//         </Link>

        
//       </div>
//     </div>
//   </div>
// </section>
// </div>
//   );
// }

import { Stars } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import React, { useEffect } from "react";
import { FiArrowRight } from "react-icons/fi";
import {
  useMotionTemplate,
  useMotionValue,
  motion,
  animate,
} from "framer-motion";

const COLORS_TOP = ["#FF4C4C", "#D72638", "#E63946", "#FF6B6B"];

export default function Home()  {
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
  const border = useMotionTemplate`1px solid ${color}`;
  const boxShadow = useMotionTemplate`0px 4px 24px ${color}`;

  return (
    <motion.section
      style={{
        backgroundImage,
      }}
      className="relative grid min-h-screen place-content-center overflow-hidden bg-gray-950 px-4 py-24 text-gray-200"
    >
      <div className="relative z-10 flex flex-col items-center">
        
        <h1 className="max-w-3xl bg-gradient-to-br from-white to-gray-400 bg-clip-text text-center text-3xl font-medium leading-tight text-transparent sm:text-5xl sm:leading-tight md:text-5xl md:leading-tight">
          DYNAMIC HACKATHON - TEAM DRONA
        </h1>
        <p className="my-6 max-w-xl text-center text-base leading-relaxed md:text-lg md:leading-relaxed">
        Dive in and discover our novel solutions assisted by computer vision driving safety
        and smooth operations at construction sites
        </p>
        <Link href="/auth">
        <motion.button
          style={{
            border,
            boxShadow,
          }}
          whileHover={{
            scale: 1.015,
          }}
          whileTap={{
            scale: 0.985,
          }}
          className="group relative flex w-fit items-center gap-1.5 rounded-full bg-gray-950/10 px-4 py-2 text-gray-50 transition-colors hover:bg-gray-950/50"
        >
          Enter Dashboard
          <FiArrowRight className="transition-transform group-hover:-rotate-45 group-active:-rotate-12" />
        </motion.button>
        </Link>
      </div>

      <div className="absolute inset-0 z-0">
        <Canvas>
          <Stars radius={50} count={2500} factor={4} fade speed={2} />
        </Canvas>
      </div>
    </motion.section>
  );
};