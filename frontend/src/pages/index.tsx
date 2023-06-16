import GridBasicExample from '../components/Form';
import HeaderComponent from '@/components/HeaderComponent';
import Results from '@/components/Results';
import { useState } from 'react';
import { ResultsProps } from '@/components/Results';

export default function Home() {

  const [result, setResult] = useState<ResultsProps["result"]>(null);
  const [waitingForResults, setWaitingForResults] = useState<boolean>(false)

  

  return (
    <main className='min-h-screen '>
      <HeaderComponent/>
      <div className="flex flex-col items-center justify-between p-24">
        <div className="z-10 w-full justify-between text-sm lg:flex text-lg">
          <div className="w-1/3">
              <GridBasicExample setResults={setResult} setWaitingForResults={setWaitingForResults}/>
            </div>
            <div className="ml-4 w-2/3 ">
              <Results result={result} waitingForResults={waitingForResults}/>
            </div>
        </div>
      </div>
    </main>
  )
}
