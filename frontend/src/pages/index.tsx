import GridBasicExample from '../components/Form';
import HeaderComponent from '@/components/HeaderComponent';
import Results from '@/components/Results';
import { useState } from 'react';

export default function Home() {

  const [result, setResult] = useState("");

  return (
    <main >
      <HeaderComponent/>
      <div className="flex min-h-screen flex-col items-center justify-between p-24">
        <div className="z-10 w-full max-w-5xl justify-between text-sm lg:flex text-lg">
          <GridBasicExample onButtonClick={setResult}/>
          <Results result={result}/>
        </div>
      </div>
    </main>
  )
}
