import {useId} from 'react'
import type {Patient, Outcome, CalibrationBin} from './types'
export const pct=(value:number|null|undefined,digits=0)=>value == null || !Number.isFinite(value) ? '—' : `${(value*100).toFixed(digits)}%`
export const fixed=(value:number|null|undefined,digits=2)=>value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
export function Anatomy({patient,outcome,compact=false}:{patient?:Patient;outcome?:Outcome;compact?:boolean}) {
  const id=useId().replace(/:/g,'')
  const angle=(outcome?.strategy.direction_deg ?? 35)*Math.PI/180
  const distance=outcome?.strategy.distance_mm ?? 16
  const start={x:263,y:229}
  const end={x:start.x+Math.cos(angle)*distance*3,y:start.y-Math.sin(angle)*distance*3}
  return <svg className={`anatomy ${compact?'compact':''}`} viewBox="0 0 480 330" role="img" aria-label={patient?`Schematic of patient ${patient.id} and the selected pull direction`:'Schematic of a gallbladder, tissue attachments and retraction direction'}>
    <defs><pattern id={`${id}grid`} width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#dce7e3" strokeWidth=".6"/></pattern><linearGradient id={`${id}tissue`} x1="0" x2="1" y1="0" y2="1"><stop stopColor="#bcd4bd"/><stop offset="1" stopColor="#6d9f87"/></linearGradient><marker id={`${id}arrow`} markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#d98162"/></marker></defs>
    <rect width="480" height="330" fill={`url(#${id}grid)`}/>
    <path d="M72 112C83 54 164 39 234 52C294 39 384 74 415 137C428 169 387 186 347 169C283 146 238 161 192 155C139 154 97 171 78 146Z" fill="#d3b09c" fillOpacity=".36" stroke="#b99e8d" strokeWidth="1.2"/>
    <path d="M221 104C213 82 188 77 172 96C144 128 160 192 196 213C224 231 236 250 258 249C276 247 269 230 254 213C248 201 258 164 250 141C246 124 230 116 221 104Z" fill={`url(#${id}tissue)`} stroke="#487766" strokeWidth="1.8"/>
    {[0,1,2,3,4].map(i=><path key={i} d={`M${182+i*13} ${111+i*5} Q${191+i*12} ${133+i*6} ${205+i*9} ${141+i*9}`} stroke="#638b73" strokeWidth=".9" fill="none" opacity=".45"/>)}
    <path d="M259 248Q268 267 277 276L299 304M277 276L287 232Q293 214 300 198" fill="none" stroke="#8eae95" strokeWidth="9" strokeLinecap="round"/>
    <path d="M264 254L278 276L287 241Z" fill="#f5d3a1" fillOpacity=".6" stroke="#bc9b67" strokeDasharray="3 3"/>
    {Array.from({length:Math.min(patient?.adhesion_count ?? 3,8)},(_,i)=><path key={i} d={`M${168+i*10} ${130+i*9}Q${147+i*10} ${116+i*9} ${134+i*10} ${125+i*8}`} fill="none" stroke="#b5a18a" strokeWidth="1.7" strokeDasharray="3 3"/>)}
    <line x1="186" y1="105" x2="186" y2="57" stroke="#487766" strokeWidth="1.5" strokeDasharray="4 3"/>
    <circle cx="186" cy="105" r="5" fill="#fafbf6" stroke="#487766" strokeWidth="2"/>
    <line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke="#d98162" strokeWidth="2.6" markerEnd={`url(#${id}arrow)`}/>
    <circle cx={start.x} cy={start.y} r="5" fill="#fff9f3" stroke="#d98162" strokeWidth="2"/>
    {!compact && <><text x="41" y="71" className="svg-label">Fundus held fixed</text><path d="M140 68H173L182 91" stroke="#81958b" fill="none"/><text x="323" y="246" className="svg-label">Neck retraction</text><path d="M314 241L285 231" stroke="#9f9a86" fill="none"/><text x="316" y="294" className="svg-label">Exposure proxy</text><path d="M312 289L291 279" stroke="#9f9a86" fill="none"/><text x="16" y="311" className="svg-note">ILLUSTRATIVE SCHEMATIC · NOT ANATOMICAL GEOMETRY</text></>}
  </svg>
}
export function ForcePlot({patient}:{patient:Patient}) {
  const times=patient.probe.times_s, forces=patient.probe.forces_n
  const max=Math.max(...forces,0.1)*1.15
  const points=forces.map((f,i)=>`${42+(times[i]??0)*310},${145-f/max*112}`).join(' ')
  return <svg viewBox="0 0 380 185" role="img" aria-label="One-second probe force response" className="chart">
    {[0,.5,1].map(t=><g key={t}><line x1="42" y1={145-t*112} x2="352" y2={145-t*112} stroke="#e3e8e5"/><text x="30" y={149-t*112} textAnchor="end">{(t*max).toFixed(1)}</text></g>)}
    <path d={`M42 145L${points}L352 145Z`} fill="#288273" fillOpacity=".08"/><polyline points={points} fill="none" stroke="#258171" strokeWidth="2.5"/>
    {[0,.5,1].map(t=><text key={t} x={42+t*310} y="164" textAnchor="middle">{t.toFixed(1)}s</text>)}<text x="43" y="20">Force (N)</text>
  </svg>
}
export function CalibrationChart({bins}:{bins:CalibrationBin[]}) {
  return <svg viewBox="0 0 350 250" role="img" aria-label="Calibration plot: predicted probability versus observed fraction" className="chart">
    {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="48" y1={200-t*163} x2="320" y2={200-t*163} stroke="#e3e8e5"/><text x="37" y={204-t*163} textAnchor="end">{pct(t)}</text><text x={48+t*272} y="220" textAnchor="middle">{pct(t)}</text></g>)}
    <line x1="48" y1="200" x2="320" y2="37" stroke="#a5b4ad" strokeDasharray="5 4"/>
    {bins.filter(b=>b.count>0).map((b,i)=><g key={i}><circle cx={48+b.mean_predicted*272} cy={200-b.observed_rate*163} r={Math.min(11,4+Math.sqrt(b.count))} fill="#287e70" fillOpacity=".8"><title>{b.count} patients · predicted {pct(b.mean_predicted)} · observed {pct(b.observed_rate)}</title></circle></g>)}
    <text x="184" y="246" textAnchor="middle">Predicted probability</text><text transform="translate(11 122) rotate(-90)" textAnchor="middle">Observed fraction</text>
  </svg>
}
