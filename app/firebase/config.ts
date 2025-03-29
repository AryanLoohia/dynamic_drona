import { initializeApp, getApps, FirebaseApp } from 'firebase/app';

const firebaseConfig = {
  apiKey: "AIzaSyDHZOmC3wqbu6oTzAyLDYjiBQ3Ck-0JXBQ",
  authDomain: "dynamic-drona.firebaseapp.com",
  projectId: "dynamic-drona",
  storageBucket: "dynamic-drona.appspot.com",
  messagingSenderId: "151272211871",
  appId: "1:151272211871:web:3a304b628182a7f060569c",
  measurementId: "G-XWPVP7GF6K"
};

// Initialize Firebase only if an instance doesn't exist
let app: FirebaseApp;

if (!getApps().length) {
  app = initializeApp(firebaseConfig);
} else {
  app = getApps()[0];
}

export default app;