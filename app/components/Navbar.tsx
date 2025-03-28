"use client";

import { useRouter } from "next/navigation";
import { getAuth, signOut } from "firebase/auth";
import { useEffect, useState } from "react";

export default function Navbar() {
  const router = useRouter();
  const [auth, setAuth] = useState<any>(null);

  useEffect(() => {
    setAuth(getAuth());
  }, []);

  const handleLogout = async () => {
    if (auth) {
      await signOut(auth);
      router.push("/auth"); // Redirect to login after logout
    }
  };

  return (
    <nav className="fixed top-0 left-0 w-full flex justify-between items-center bg-zinc-800 text-white py-3 px-6">
      <button
        onClick={() => router.push("/")}
        className="bg-red-500 px-4 py-2 rounded hover:bg-red-700"
      >
        Back to Home
      </button>
      <button
        onClick={() => router.push("/dashboard")}
        className="bg-red-500 px-4 py-2 rounded hover:bg-red-700"
      >
        Dashboard
      </button>
      
      <button
        onClick={handleLogout}
        className="bg-red-500 px-4 py-2 rounded hover:bg-red-700"
      >
        Logout
      </button>
    </nav>
  );
}
