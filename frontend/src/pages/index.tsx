import GridBasicExample from '../components/Form';
import HeaderComponent from '@/components/HeaderComponent';
import Results from '@/components/Results';
import { useState } from 'react';
import { ResultsProps } from '@/components/Results';

export default function Home() {

  const [result, setResult] = useState<ResultsProps["result"]>(null);

  return (
    <main >
      <HeaderComponent/>
      <div className="flex min-h-screen flex-col items-center justify-between p-24">
        <div className="z-10 w-full max-w-5xl justify-between text-sm lg:flex text-lg">
          <div className="w-2/5">
              <GridBasicExample onButtonClick={setResult}/>
            </div>
            <div className="ml-4 w-3/5 ">
              <Results result={result}/>
            </div>
        </div>
      </div>
    </main>
  )
}
