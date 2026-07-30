import ReactDOM from 'react-dom/client';
import { FaceLivenessDetector } from '@aws-amplify/ui-react-liveness';
import '@aws-amplify/ui-react/styles.css';

let _root = null;

const DISPLAY_TEXT = {
  hintCenterFaceText:           'Place face in the circle',
  hintFaceOffCenterText:        'Center your face',
  hintTooFarText:               '📷 Move closer',
  hintTooCloseText:             '↔ Move back a little',
  hintHoldFaceForFreshnessText: 'Hold still',
  hintConnectingText:           'Connecting…',
  hintVerifyingText:            'Verifying…',
  hintTooManyFacesText:         'One face only',
  hintCanNotIdentifyText:       'Face not detected — try again',
  hintIlluminationTooBrightText:'Too bright — move to shade',
  hintIlluminationTooDarkText:  'Too dark — find better light',
  hintFaceDetectedText:         'Hold still…',
  cancelLivenessCheckText:      '✕',
  recordingIndicatorText:       '● REC',
};

function LivenessChallenge({ sessionId, region, credentials, onSuccess, onError }) {
  return (
    <FaceLivenessDetector
      sessionId={sessionId}
      region={region}
      onAnalysisComplete={onSuccess}
      onError={(err) => {
        console.error('[AWS Liveness]', err);
        onError(err);
      }}
      disableStartScreen={true}
      displayText={DISPLAY_TEXT}
      config={{
        credentialProvider: async () => ({
          accessKeyId:     credentials.AccessKeyId,
          secretAccessKey: credentials.SecretAccessKey,
          sessionToken:    credentials.SessionToken,
        }),
      }}
    />
  );
}

export function mount(containerId, props) {
  const el = document.getElementById(containerId);
  if (!el) throw new Error('LivenessWidget: container not found: ' + containerId);
  _root = ReactDOM.createRoot(el);
  _root.render(<LivenessChallenge {...props} />);
}

export function unmount() {
  if (_root) { _root.unmount(); _root = null; }
}
